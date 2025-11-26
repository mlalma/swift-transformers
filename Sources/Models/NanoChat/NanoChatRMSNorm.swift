import Foundation
import MLX
import MLXNN

final class NanoChatRMSNorm: Module {
    let eps: Float
    
    init(eps: Float = 1e-6) {
        self.eps = eps
        super.init()
    }

    /// Compute RMS normalization: x * rsqrt(mean(x^2) + eps)
    private func norm(_ x: MLXArray) -> MLXArray {
        let variance = pow(x, 2).mean(axes: [-1], keepDims: true)
        return x * rsqrt(variance + eps)
    }

    /// Apply normalization preserving the original type
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let normalized = norm(x.asType(.float32))
        return normalized.asType(x.dtype)
    }

    var description: String {
        return "NanoChatRMSNorm(eps=\(eps))"
    }
}
