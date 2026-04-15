#![cfg_attr(all(not(feature = "std"), not(test)), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]

#[cfg(all(not(feature = "std"), not(feature = "libm")))]
compile_error!("earshot's `libm` feature must be enabled when the `std` feature is disabled");

#[cfg(feature = "alloc")]
extern crate alloc;

use core::{f32, ptr};

mod default_predictor;
mod fft;
mod filters;
mod util;

pub use self::default_predictor::DefaultPredictor;
use self::util::libm;

/// Used by [`Detector`] to predict the VAD score of a frame based on extracted features.
///
/// # Stability
/// If you wish to implement `Predictor` yourself, note that **the API is unstable and subject to change!**
pub trait Predictor {
	#[doc(hidden)]
	fn reset(&mut self);

	#[doc(hidden)]
	fn normalize(&self, features: &mut [f32]);

	#[doc(hidden)]
	fn predict(&mut self, features: &[f32], buffer: &mut [f32]) -> f32;
}

const FFT_SIZE: usize = 1024;
const WINDOW_SIZE: usize = 768;
const N_MELS: usize = 40;
const N_FEATURES: usize = N_MELS;
const N_CONTEXT_FRAMES: usize = 3;
const N_BINS: usize = FFT_SIZE / 2 + 1;
const PRE_EMPHASIS_COEFF: f32 = 0.97;
const POWER_FAC: f32 = 1. / (32768.0f32 * 32768.0);

/// A voice activity detector. Create one per separate audio stream.
///
/// # Stack size
/// `Detector` is a fairly large object, as it allocates its state (about 8 KiB by default) on the stack. If stack space
/// is a concern, use a `Box<Detector>` instead. (Maps or vectors of `Detector`s shouldn't need to worry about this).
pub struct Detector<P = DefaultPredictor> {
	predictor: P,
	prev_signal: f32,
	sample_ring_buffer: [f32; 768],
	features: [f32; N_FEATURES * N_CONTEXT_FRAMES],
	buffer: [f32; 1026]
}

impl Default for Detector<DefaultPredictor> {
	fn default() -> Self {
		Self::new(DefaultPredictor::new())
	}
}

impl Detector<DefaultPredictor> {
	pub const fn const_default() -> Detector<DefaultPredictor> {
		Self::new(DefaultPredictor::new())
	}

	/// Creates a new `Detector` directly on the heap, without ever allocating the large amount of stack space that
	/// `Detector` normally uses.
	///
	/// This is preferred over `Box::<Detector>::default()`, since that creates the detector on the stack before moving
	/// it to the heap.
	#[cfg(feature = "alloc")]
	#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
	pub fn default_boxed() -> Box<Self> {
		// TODO: use new_zeroed instead, MSRV 1.92
		let mut boxed = alloc::boxed::Box::<Self>::new_uninit();
		let mut detector = unsafe {
			let boxed_ptr = boxed.as_mut_ptr();
			core::ptr::write(&raw mut (*boxed_ptr).predictor, DefaultPredictor::new());
			boxed.assume_init()
		};
		detector.prev_signal = 0.0;
		detector.sample_ring_buffer.fill(0.0);
		detector.features.fill(0.0);
		detector.buffer.fill(0.0);
		detector
	}
}

impl<P: Predictor> Detector<P> {
	/// Creates a new `Detector` on the stack.
	///
	/// To create directly on the heap, see [`Detector::new_boxed`] instead.
	pub const fn new(predictor: P) -> Self {
		Self {
			predictor,
			prev_signal: 0.0,
			sample_ring_buffer: [0.0; 768],
			features: [0.0; N_FEATURES * N_CONTEXT_FRAMES],
			buffer: [0.0; 1026]
		}
	}

	/// Creates a new `Detector` directly on the heap, without ever allocating the large amount of stack space that
	/// `Detector` normally uses.
	///
	/// This is preferred over `Box::new(Detector::new(predictor))`, since that creates the detector on the stack before
	/// moving it to the heap.
	#[cfg(feature = "alloc")]
	#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
	pub fn new_boxed(predictor: P) -> Box<Self> {
		// TODO: use new_zeroed instead, MSRV 1.92
		let mut boxed = alloc::boxed::Box::<Self>::new_uninit();
		let mut detector = unsafe {
			let boxed_ptr = boxed.as_mut_ptr();
			core::ptr::write(&raw mut (*boxed_ptr).predictor, predictor);
			boxed.assume_init()
		};
		detector.prev_signal = 0.0;
		detector.sample_ring_buffer.fill(0.0);
		detector.features.fill(0.0);
		detector.buffer.fill(0.0);
		detector
	}

	/// Resets the internal state of the voice activity detector.
	///
	/// The detector should be reset whenever:
	/// - the recording device changes; or
	/// - the detector is being used for a new audio sequence.
	pub fn reset(&mut self) {
		self.predictor.reset();
		self.prev_signal = 0.0;
		self.sample_ring_buffer.fill(0.0);
		self.features.fill(0.0);
	}

	/// Predicts the voice activity score of a single input frame of 16-bit PCM audio.
	///
	/// The frame:
	/// - should be sampled at 16 KHz;
	/// - should be exactly 256 samples (so 16 ms) in length.
	///
	/// The output score is between `[0, 1]`. Scores over 0.5 can generally be considered voice, but the exact threshold
	/// can be adjusted according to application-specific needs.
	pub fn predict_i16(&mut self, frame: &[i16]) -> f32 {
		debug_assert_eq!(frame.len(), 256, "frame should be exactly 256 samples");
		if frame.len() != 256 {
			return -1.0;
		}

		unsafe {
			ptr::copy(self.sample_ring_buffer.as_ptr().add(256), self.sample_ring_buffer.as_mut_ptr(), 512);
		};
		for (emph, sample) in (&mut self.sample_ring_buffer[512..]).iter_mut().zip(frame.iter()) {
			let sample = *sample as f32;
			*emph = sample - PRE_EMPHASIS_COEFF * self.prev_signal;
			self.prev_signal = sample;
		}

		self.predict_inner()
	}

	/// Predicts the voice activity score of a single input frame of 32-bit floating-point PCM audio.
	///
	/// The frame:
	/// - should be sampled at 16 KHz;
	/// - should be exactly 256 samples (so 16 ms) in length;
	/// - should consist only of samples in the range [-1, 1].
	///
	/// The output score is between `[0, 1]`. Scores over 0.5 can generally be considered voice, but the exact threshold
	/// can be adjusted according to application-specific needs.
	pub fn predict_f32(&mut self, frame: &[f32]) -> f32 {
		debug_assert_eq!(frame.len(), 256, "frame should be exactly 256 samples");
		if frame.len() != 256 {
			return -1.0;
		}

		debug_assert!(
			*frame
				.iter()
				.max_by(|x, y| x.abs().partial_cmp(&y.abs()).unwrap_or(core::cmp::Ordering::Equal))
				.unwrap() <= 1.0,
			"input frame should be in the range [-1, 1]"
		);

		/// We perform the FFT at i16 scale and scale down afterwards; doing the FFT at f32 scale ([-1, 1]) loses a lot
		/// of precision.
		const SCALE: f32 = 32768.0;

		unsafe {
			ptr::copy(self.sample_ring_buffer.as_ptr().add(256), self.sample_ring_buffer.as_mut_ptr(), 512);
		};
		for (emph, sample) in (&mut self.sample_ring_buffer[512..]).iter_mut().zip(frame.iter()) {
			let sample = *sample * SCALE;
			*emph = sample - PRE_EMPHASIS_COEFF * self.prev_signal;
			self.prev_signal = sample;
		}

		self.predict_inner()
	}

	fn predict_inner(&mut self) -> f32 {
		// windowize for FFT
		for i in 0..WINDOW_SIZE {
			self.buffer[i] = self.sample_ring_buffer[i] * filters::HANN_WINDOW[i];
		}
		// FFT size is 1024 but window size is 768, so fill the rest with zeros (+2 to store nyquist frequency)
		unsafe {
			ptr::write_bytes(self.buffer.as_mut_ptr().add(WINDOW_SIZE), 0, 256 + 2);
		};

		fft::rfft_1024(&mut self.buffer);
		for i in 0..N_BINS {
			let j = i * 2;
			self.buffer[i] = fft::Complex32::new(self.buffer[j], self.buffer[j + 1]).norm_sqr()
				// downscale from i16 scale
				* POWER_FAC;
		}

		unsafe {
			ptr::copy(self.features.as_ptr().add(N_FEATURES), self.features.as_mut_ptr(), N_FEATURES * (N_CONTEXT_FRAMES - 1));
		};
		let cur_frame_features = &mut self.features[(N_FEATURES * (N_CONTEXT_FRAMES - 1))..];
		for i in 0..N_MELS {
			let mut per_band_value = 0.;
			let (start, coeffs) = filters::MEL_COEFFS[i];
			for (offs, coeff) in coeffs.iter().enumerate() {
				per_band_value += self.buffer[start + offs] * *coeff;
			}

			cur_frame_features[i] = libm::logf(per_band_value + 1e-20);
		}
		self.predictor.normalize(cur_frame_features);

		self.predictor.predict(&self.features, &mut self.buffer)
	}
}
