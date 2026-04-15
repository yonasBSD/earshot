#[cfg(feature = "libm")]
pub use libm;
#[cfg(not(feature = "libm"))]
pub mod libm {
	#[inline(always)]
	pub fn sqrtf(f: f32) -> f32 {
		f.sqrt()
	}
	#[inline(always)]
	pub fn logf(f: f32) -> f32 {
		f.ln()
	}
	#[inline(always)]
	pub fn expf(f: f32) -> f32 {
		f.exp()
	}
}
