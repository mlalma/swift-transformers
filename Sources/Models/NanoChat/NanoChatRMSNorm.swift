import Foundation
import MLX
import MLXNN

final class NanoChatRMSNorm: Module {
    let eps: Float
    
    init(eps: Float = 1e-6) {
        self.eps = eps
        super.init()
    }
    
    private func norm(_ x: MLXArray) -> MLXArray {
        // Compute RMS normalization: x * rsqrt(mean(x^2) + eps)
        let variance = pow(x, 2).mean(axes: [-1], keepDims: true)
        return x * rsqrt(variance + eps)
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Apply normalization preserving the original type
        let normalized = norm(x.asType(.float32))
        return normalized.asType(x.dtype)
    }

    var description: String {
        return "NanoChatRMSNorm(eps=\(eps))"
    }
}
