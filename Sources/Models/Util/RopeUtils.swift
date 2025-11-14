import Foundation
import MLX

/// Errors that can occur during RoPE computation.
public enum RopeError: Error, CustomStringConvertible {
    case missingRopeParameters
    case missingParameter(String)
    case unsupportedRopeType(RopeType)
    case invalidConfiguration(String)
    
    public var description: String {
        switch self {
        case .missingRopeParameters:
            return "RoPE parameters are missing from configuration"
        case .missingParameter(let param):
            return "Missing required RoPE parameter: \(param)"
        case .unsupportedRopeType(let type):
            return "Unsupported RoPE type: \(type)"
        case .invalidConfiguration(let message):
            return "Invalid RoPE configuration: \(message)"
        }
    }
}

/// Namepace for rotary positional encoding utility methods to help the LLM implementations.
enum RopeUtils {}

extension RopeUtils {
    /// Computes RoPE parameters using the specified RoPE type.
    /// - Parameters:
    ///   - config: The model configuration.
    ///   - ropeType: The type of RoPE scaling to use.
    ///   - seqLen: Optional sequence length for dynamic RoPE types.
    /// - Returns: A tuple of (inverse frequencies, attention scaling factor).
    static func computeRopeParameters(
        config: NanoChatConfig,
        ropeType: RopeType,
        seqLen: Int? = nil
    ) throws -> (MLXArray, Float) {
        switch ropeType {
        case .default:
            return try computeDefaultRopeParameters(config: config, seqLen: seqLen)
        case .linear:
            return try computeLinearScalingRopeParameters(config: config, seqLen: seqLen)
        case .dynamic:
            return try computeDynamicNTKParameters(config: config, seqLen: seqLen)
        case .yarn:
            return try computeYarnParameters(config: config, seqLen: seqLen)
        /*
        case .longrope:
            return try computeLongRopeParameters(config: config, seqLen: seqLen)
        case .llama3:
            return try computeLlama3Parameters(config: config, seqLen: seqLen)
        */
        default:
            throw RopeError.unsupportedRopeType(ropeType)
        }
    }
    
    /// Computes the inverse frequencies according to the original RoPE implementation.
    /// - Parameters:
    ///   - config: The model configuration.
    ///   - seqLen: The current sequence length (unused for default RoPE).
    /// - Returns: A tuple of (inverse frequencies, attention scaling factor).
    static func computeDefaultRopeParameters(
        config: NanoChatConfig,
        seqLen: Int? = nil
    ) throws -> (MLXArray, Float) {
        guard let ropeParams = config.ropeParameters else {
            throw RopeError.missingRopeParameters
        }
        
        let base = ropeParams.ropeTheta
        let dim = config.hiddenSize / config.numAttentionHeads
        // Unused in default RoPE
        let attentionFactor: Float = 1.0
        
        // Compute inverse frequencies: 1.0 / (base^(i/dim)) for i in [0, 2, 4, ..., dim-2]
        let indices = MLXArray(stride(from: 0, to: dim, by: 2).map { Float($0) })
        let exponents = indices / Float(dim)
        let invFreq = 1.0 / MLX.pow(MLXArray(Float(base)), exponents)
        
        return (invFreq, attentionFactor)
    }
    
    /// Computes inverse frequencies with linear scaling.
    /// Credits to Reddit user /u/kaiokendev for the linear scaling approach.
    static func computeLinearScalingRopeParameters(
        config: NanoChatConfig,
        seqLen: Int? = nil
    ) throws -> (MLXArray, Float) {
        guard let ropeParams = config.ropeParameters else {
            throw RopeError.missingRopeParameters
        }
        
        guard let factor = ropeParams.factor else {
            throw RopeError.missingParameter("factor")
        }
        
        let base = ropeParams.ropeTheta
        let partialRotaryFactor = config.partialRotaryFactor ?? 1.0
        let headDim = config.headDim ?? (config.hiddenSize / config.numAttentionHeads)
        let dim = Int(Double(headDim) * partialRotaryFactor)
        let attentionFactor: Float = 1.0
        
        // Compute the inverse frequencies
        let indices = MLXArray(stride(from: 0, to: dim, by: 2).map { Float($0) })
        let exponents = indices / Float(dim)
        var invFreq = 1.0 / MLX.pow(MLXArray(Float(base)), exponents)
        
        // Apply linear scaling to the frequencies
        // *NOTE*: originally, scaling was applied to the position_ids. However, we get `embs = inv_freq @ position_ids`, so
        // applying scaling to the inverse frequencies is equivalent.
        invFreq = invFreq / Float(factor)
        
        return (invFreq, attentionFactor)
    }
    
    /// Computes inverse frequencies with NTK scaling.
    static func computeDynamicNTKParameters(
        config: NanoChatConfig,
        seqLen: Int? = nil
    ) throws -> (MLXArray, Float) {
        guard let ropeParams = config.ropeParameters else {
            throw RopeError.missingRopeParameters
        }
        
        guard let factor = ropeParams.factor else {
            throw RopeError.missingParameter("factor")
        }
        
        var base = ropeParams.ropeTheta
        let dim = config.hiddenSize / config.numAttentionHeads
        let maxPositionEmbeddings = config.maxPositionEmbeddings
        let attentionFactor: Float = 1.0
        
        // Determine sequence length
        let effectiveSeqLen = seqLen ?? maxPositionEmbeddings
        let seqLength = max(effectiveSeqLen, maxPositionEmbeddings)
        
        // Compute dynamic base
        let scaling = (Double(factor) * Double(seqLength) / Double(maxPositionEmbeddings)) - (Double(factor) - 1.0)
        let exponent = Double(dim) / Double(dim - 2)
        base = base * pow(scaling, exponent)
        
        // Compute inverse frequencies
        let indices = MLXArray(stride(from: 0, to: dim, by: 2).map { Float($0) })
        let exponents = indices / Float(dim)
        let invFreq = 1.0 / MLX.pow(MLXArray(Float(base)), exponents)
        
        return (invFreq, attentionFactor)
    }
    
    /// Computes inverse frequencies with Yarn scaling.
    /// Reference: https://huggingface.co/papers/2309.00071
    static func computeYarnParameters(
        config: NanoChatConfig,
        seqLen: Int? = nil
    ) throws -> (MLXArray, Float) {
        guard let ropeParams = config.ropeParameters else {
            throw RopeError.missingRopeParameters
        }
        
        let base = ropeParams.ropeTheta
        let partialRotaryFactor = config.partialRotaryFactor ?? 1.0
        let headDim = config.headDim ?? (config.hiddenSize / config.numAttentionHeads)
        let dim = Int(Double(headDim) * partialRotaryFactor)
        
        guard var factor = ropeParams.factor else {
            throw RopeError.missingParameter("factor")
        }
        
        var attentionFactor = ropeParams.attentionFactor
        let mscale = ropeParams.mscale
        let mscaleAllDim = ropeParams.mscaleAllDim
        
        // NOTE: DeepSeek-V3 (and potentially other models) modify `max_position_embeddings` and have a
        // `original_max_position_embeddings` field containing the pretrained value. They use the ratio between these two
        // values to compute the default attention scaling factor, instead of using `factor`.
        let originalMaxPositionEmbeddings: Int
        if let original = ropeParams.originalMaxPositionEmbeddings {
            originalMaxPositionEmbeddings = original
            factor = Double(config.maxPositionEmbeddings) / Double(original)
        } else {
            originalMaxPositionEmbeddings = config.maxPositionEmbeddings
        }
        
        // Helper function to compute mscale
        func getMscale(_ scale: Double, mscale: Double = 1.0) -> Double {
            scale <= 1 ? 1.0 : (0.1 * mscale * log(scale) + 1.0)
        }
        
        // Sets the attention factor as suggested in the paper
        if attentionFactor == nil {
            if let mscaleValue = mscale, let mscaleAllDimValue = mscaleAllDim {
                attentionFactor = getMscale(factor, mscale: mscaleValue) / getMscale(factor, mscale: mscaleAllDimValue)
            } else if let mscaleValue = mscale {
                attentionFactor = getMscale(factor, mscale: mscaleValue)
            } else {
                attentionFactor = getMscale(factor)
            }
        }
        
        // Optional config options
        // beta_fast/beta_slow: as suggested in the paper, default to 32/1 (correspondingly)
        let betaFast = ropeParams.betaFast ?? 32.0
        let betaSlow = ropeParams.betaSlow ?? 1.0
        
        // Helper function: Inverse dimension formula to find the dimension based on the number of rotations
        func findCorrectionDim(numRotations: Double, dim: Int, base: Double, maxPositionEmbeddings: Int) -> Double {
            return (Double(dim) * log(Double(maxPositionEmbeddings) / (numRotations * 2.0 * Double.pi))) / (2.0 * log(base))
        }
        
        // Helper function: Find dimension range bounds based on rotations
        func findCorrectionRange(lowRot: Double, highRot: Double, dim: Int, base: Double, maxPositionEmbeddings: Int, truncate: Bool) -> (Double, Double) {
            var low = findCorrectionDim(numRotations: lowRot, dim: dim, base: base, maxPositionEmbeddings: maxPositionEmbeddings)
            var high = findCorrectionDim(numRotations: highRot, dim: dim, base: base, maxPositionEmbeddings: maxPositionEmbeddings)
            
            if truncate {
                low = floor(low)
                high = ceil(high)
            }
            
            return (max(low, 0), min(high, Double(dim - 1)))
        }
        
        // Helper function: Linear ramp factor
        func linearRampFactor(min: Double, max: Double, dim: Int) -> MLXArray {
            let maxValue = max + (min == max ? 0.001 : 0.0)
            let indices = MLXArray((0..<dim).map { Float($0) })
            let linearFunc = (indices - Float(min)) / Float(maxValue - min)
            let rampFunc = MLX.clip(linearFunc, min: MLXArray(0.0), max: MLXArray(1.0))
            return rampFunc
        }
        
        // Note on variable naming: "interpolation" comes from the original technique, where we interpolate the position IDs
        // to expand the possible context length. In other words, interpolation = apply scaling factor.
        let indices = MLXArray(stride(from: 0, to: dim, by: 2).map { Float($0) })
        let exponents = indices / Float(dim)
        let posFreqs = MLX.pow(MLXArray(Float(base)), exponents)
        let invFreqExtrapolation = 1.0 / posFreqs
        let invFreqInterpolation = 1.0 / (Float(factor) * posFreqs)
        
        let truncate = ropeParams.truncate ?? true
        let (low, high) = findCorrectionRange(
            lowRot: betaFast,
            highRot: betaSlow,
            dim: dim,
            base: base,
            maxPositionEmbeddings: originalMaxPositionEmbeddings,
            truncate: truncate
        )
        
        // Get n-dimensional rotational scaling corrected for extrapolation
        let invFreqExtrapolationFactor = 1.0 - linearRampFactor(min: low, max: high, dim: dim / 2)
        let invFreq = invFreqInterpolation * (1.0 - invFreqExtrapolationFactor) + invFreqExtrapolation * invFreqExtrapolationFactor
        
        return (invFreq, Float(attentionFactor ?? 1.0))
    }
    
    /*
    /// Computes inverse frequencies with LongRoPE scaling.
    /// Reference: https://github.com/microsoft/LongRoPE
    static func computeLongRopeParameters(
        config: NanoChatConfig,
        seqLen: Int? = nil
    ) throws -> (MLXArray, Float) {
        guard let ropeParams = config.ropeParameters else {
            throw RopeError.missingRopeParameters
        }
        
        guard let shortFactor = ropeParams.shortFactor else {
            throw RopeError.missingParameter("short_factor")
        }
        
        guard let longFactor = ropeParams.longFactor else {
            throw RopeError.missingParameter("long_factor")
        }
        
        let base = ropeParams.ropeTheta
        let partialRotaryFactor = config.partialRotaryFactor ?? 1.0
        let headDim = config.headDim ?? (config.hiddenSize / config.numAttentionHeads)
        let dim = Int(Double(headDim) * partialRotaryFactor)
        let originalMaxPosEmbeddings = ropeParams.originalMaxPositionEmbeddings ?? config.maxPositionEmbeddings
        
        // Determine which factors to use based on sequence length
        let extFactors: [Double]
        if let seqLen = seqLen, seqLen > originalMaxPosEmbeddings {
            extFactors = longFactor
        } else {
            extFactors = shortFactor
        }
        
        // Compute attention factor
        let factor = ropeParams.factor ?? 1.0
        let attentionFactor: Float
        if let explicit = ropeParams.attentionFactor {
            attentionFactor = Float(explicit)
        } else if factor <= 1.0 {
            attentionFactor = 1.0
        } else {
            attentionFactor = sqrt(1.0 + log(Float(factor)) / log(Float(originalMaxPosEmbeddings)))
        }
        
        // Compute inverse frequencies with extension factors
        let indices = MLXArray(stride(from: 0, to: dim, by: 2).map { Float($0) })
        let invFreqShape = indices / Float(dim)
        let extFactorsArray = MLXArray(extFactors.map { Float($0) })
        let invFreq = 1.0 / (extFactorsArray * MLX.pow(MLXArray(Float(base)), invFreqShape))
        
        return (invFreq, attentionFactor)
    }
    
    /// Computes inverse frequencies for Llama 3.1 style RoPE.
    static func computeLlama3Parameters(
        config: NanoChatConfig,
        seqLen: Int? = nil
    ) throws -> (MLXArray, Float) {
        guard let ropeParams = config.ropeParameters else {
            throw RopeError.missingRopeParameters
        }
        
        guard let factor = ropeParams.factor else {
            throw RopeError.missingParameter("factor")
        }
        
        guard let lowFreqFactor = ropeParams.lowFreqFactor else {
            throw RopeError.missingParameter("low_freq_factor")
        }
        
        guard let highFreqFactor = ropeParams.highFreqFactor else {
            throw RopeError.missingParameter("high_freq_factor")
        }
        
        guard let originalMaxPosEmbeddings = ropeParams.originalMaxPositionEmbeddings else {
            throw RopeError.missingParameter("original_max_position_embeddings")
        }
        
        let base = ropeParams.ropeTheta
        let partialRotaryFactor = config.partialRotaryFactor ?? 1.0
        let headDim = config.headDim ?? (config.hiddenSize / config.numAttentionHeads)
        let dim = Int(Double(headDim) * partialRotaryFactor)
        let attentionFactor: Float = 1.0
        
        // Compute base inverse frequencies
        let indices = MLXArray(stride(from: 0, to: dim, by: 2).map { Float($0) })
        let exponents = indices / Float(dim)
        let invFreq = 1.0 / MLX.pow(MLXArray(Float(base)), exponents)
        
        let lowFreqWavelen = Float(originalMaxPosEmbeddings) / Float(lowFreqFactor)
        let highFreqWavelen = Float(originalMaxPosEmbeddings) / Float(highFreqFactor)
        
        // Compute wavelengths
        let wavelen = 2.0 * Float.pi / invFreq
        
        // Apply Llama3 scaling logic
        let invFreqLlama = MLX.where_(
            wavelen .> lowFreqWavelen,
            invFreq / Float(factor),
            invFreq
        )
        
        // Smooth factor for medium frequencies
        let smoothFactor = (Float(originalMaxPosEmbeddings) / wavelen - Float(lowFreqFactor)) /
                          (Float(highFreqFactor) - Float(lowFreqFactor))
        let smoothedInvFreq = (1.0 - smoothFactor) * invFreqLlama / Float(factor) +
                             smoothFactor * invFreqLlama
        
        let isMediumFreq = MLX.logicalAnd(
            wavelen .>= highFreqWavelen,
            wavelen .<= lowFreqWavelen
        )
        
        let finalInvFreq = MLX.where_(isMediumFreq, smoothedInvFreq, invFreqLlama)
        
        return (finalInvFreq, attentionFactor)
    }
    */
}
