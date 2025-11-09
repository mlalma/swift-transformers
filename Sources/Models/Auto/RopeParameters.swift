import Foundation
import Hub

/// RoPE type variants supported by the framework.
enum RopeType: String, Codable, CaseIterable {
    /// Original RoPE implementation
    case `default`

    /// Linear scaling RoPE
    case linear

    /// Dynamic NTK scaling RoPE
    case dynamic

    /// YaRN (Yet another RoPE extensioN)
    case yarn

    /// Long RoPE for extended context
    case longrope

    /// Llama 3 specific RoPE implementation
    case llama3
}

/// RoPE (Rotary Position Embedding) parameters for transformer models.
/// This struct contains configuration for different RoPE supported variants.
struct RopeParameters: Codable, Equatable, CustomStringConvertible {
    /// The base period of the RoPE embeddings.
    let ropeTheta: Double

    /// The sub-variant of RoPE to use.
    let ropeType: RopeType

    /// The scaling factor to apply to the RoPE embeddings.
    /// Used with all rope types except 'default'.
    let factor: Double?

    /// The original max position embeddings used during pretraining.
    /// Used with 'dynamic', 'longrope' and 'llama3'.
    let originalMaxPositionEmbeddings: Int?

    /// The scaling factor to be applied on the attention computation.
    /// Used with 'yarn' and 'longrope'.
    let attentionFactor: Double?

    /// Parameter to set the boundary for extrapolation in the linear ramp function.
    /// Only used with 'yarn'. Defaults to 32 if unspecified.
    let betaFast: Double?

    /// Parameter to set the boundary for interpolation in the linear ramp function.
    /// Only used with 'yarn'. Defaults to 1 if unspecified.
    let betaSlow: Double?

    /// The scaling factor to be applied to short contexts.
    /// Only used with 'longrope'.
    let shortFactor: [Double]?

    /// The scaling factor to be applied to long contexts.
    /// Only used with 'longrope'.
    let longFactor: [Double]?

    /// Scaling factor applied to low frequency components of the RoPE.
    /// Only used with 'llama3'.
    let lowFreqFactor: Double?

    /// Scaling factor applied to high frequency components of the RoPE.
    /// Only used with 'llama3'.
    let highFreqFactor: Double?

    // MARK: - Initialization

    /// Initialize RoPE parameters.
    ///
    /// - Parameters:
    ///   - ropeTheta: The base period of the RoPE embeddings.
    ///   - ropeType: The RoPE variant type (defaults to .default).
    ///   - factor: Scaling factor for RoPE embeddings.
    ///   - originalMaxPositionEmbeddings: Original max position embeddings from pretraining.
    ///   - attentionFactor: Scaling factor for attention computation.
    ///   - betaFast: Boundary for extrapolation (yarn only).
    ///   - betaSlow: Boundary for interpolation (yarn only).
    ///   - shortFactor: Scaling for short contexts (longrope only).
    ///   - longFactor: Scaling for long contexts (longrope only).
    ///   - lowFreqFactor: Low frequency scaling (llama3 only).
    ///   - highFreqFactor: High frequency scaling (llama3 only).
    init(
        ropeTheta: Double,
        ropeType: RopeType = .default,
        factor: Double? = nil,
        originalMaxPositionEmbeddings: Int? = nil,
        attentionFactor: Double? = nil,
        betaFast: Double? = nil,
        betaSlow: Double? = nil,
        shortFactor: [Double]? = nil,
        longFactor: [Double]? = nil,
        lowFreqFactor: Double? = nil,
        highFreqFactor: Double? = nil
    ) {
        self.ropeTheta = ropeTheta
        self.ropeType = ropeType
        self.factor = factor
        self.originalMaxPositionEmbeddings = originalMaxPositionEmbeddings
        self.attentionFactor = attentionFactor
        self.betaFast = betaFast
        self.betaSlow = betaSlow
        self.shortFactor = shortFactor
        self.longFactor = longFactor
        self.lowFreqFactor = lowFreqFactor
        self.highFreqFactor = highFreqFactor
    }

    init(fromConfig config: Config) {
        ropeTheta = config["rope_theta", Double.self] ?? 0.0
        ropeType = .init(rawValue: config["rope_type", String.self] ?? "default") ?? .default
        factor = config["factor", Double.self]
        originalMaxPositionEmbeddings = config["original_max_position_embeddings", Int.self]
        attentionFactor = config["attention_factor", Double.self]
        betaFast = config["beta_fast", Double.self]
        betaSlow = config["beta_slow", Double.self]
        shortFactor = config["short_factor", [Double].self]
        longFactor = config["long_factor", [Double].self]
        lowFreqFactor = config["low_freq_factor", Double.self]
        highFreqFactor = config["high_freq_factor", Double.self]
    }

    // MARK: - Codable

    enum CodingKeys: String, CodingKey {
        case ropeTheta = "rope_theta"
        case ropeType = "rope_type"
        case factor
        case originalMaxPositionEmbeddings = "original_max_position_embeddings"
        case attentionFactor = "attention_factor"
        case betaFast = "beta_fast"
        case betaSlow = "beta_slow"
        case shortFactor = "short_factor"
        case longFactor = "long_factor"
        case lowFreqFactor = "low_freq_factor"
        case highFreqFactor = "high_freq_factor"
    }

    // MARK: - Methods

    /// Converts the RoPE parameters to a dictionary representation.
    ///
    /// - Returns: A dictionary containing all non-nil parameters.
    func toDictionary() -> [String: Any] {
        var dict: [String: Any] = [
            "rope_theta": ropeTheta,
            "rope_type": ropeType.rawValue,
        ]

        if let factor = factor {
            dict["factor"] = factor
        }
        if let originalMaxPositionEmbeddings = originalMaxPositionEmbeddings {
            dict["original_max_position_embeddings"] = originalMaxPositionEmbeddings
        }
        if let attentionFactor = attentionFactor {
            dict["attention_factor"] = attentionFactor
        }
        if let betaFast = betaFast {
            dict["beta_fast"] = betaFast
        }
        if let betaSlow = betaSlow {
            dict["beta_slow"] = betaSlow
        }
        if let shortFactor = shortFactor {
            dict["short_factor"] = shortFactor
        }
        if let longFactor = longFactor {
            dict["long_factor"] = longFactor
        }
        if let lowFreqFactor = lowFreqFactor {
            dict["low_freq_factor"] = lowFreqFactor
        }
        if let highFreqFactor = highFreqFactor {
            dict["high_freq_factor"] = highFreqFactor
        }

        return dict
    }

    /// Creates RoPE parameters from a dictionary.
    ///
    /// - Parameter dictionary: Dictionary containing RoPE parameter values.
    /// - Returns: A RopeParameters instance if the dictionary contains valid data, nil otherwise.
    static func fromDictionary(_ dictionary: [String: Any]) -> RopeParameters? {
        guard let ropeTheta = dictionary["rope_theta"] as? Double else {
            return nil
        }

        let ropeType: RopeType
        if let ropeTypeString = dictionary["rope_type"] as? String,
           let type = RopeType(rawValue: ropeTypeString)
        {
            ropeType = type
        } else {
            ropeType = .default
        }

        return RopeParameters(
            ropeTheta: ropeTheta,
            ropeType: ropeType,
            factor: dictionary["factor"] as? Double,
            originalMaxPositionEmbeddings: dictionary["original_max_position_embeddings"] as? Int,
            attentionFactor: dictionary["attention_factor"] as? Double,
            betaFast: dictionary["beta_fast"] as? Double,
            betaSlow: dictionary["beta_slow"] as? Double,
            shortFactor: dictionary["short_factor"] as? [Double],
            longFactor: dictionary["long_factor"] as? [Double],
            lowFreqFactor: dictionary["low_freq_factor"] as? Double,
            highFreqFactor: dictionary["high_freq_factor"] as? Double
        )
    }

    /// Validates the RoPE parameters based on the RoPE type.
    ///
    /// - Throws: ValidationError if the parameters are invalid for the specified RoPE type.
    func validate() throws {
        // Type-specific validation
        switch ropeType {
        case .linear:
            try validateLinearParameters()
        case .dynamic:
            try validateDynamicParameters()
        case .yarn:
            try validateYarnParameters()
        case .longrope:
            try validateLongRopeParameters()
        case .llama3:
            try validateLlama3Parameters()
        case .default:
            // Default type only requires rope_theta
            break
        }
    }

    // MARK: - Private Validation Methods

    private func validateLinearParameters() throws {
        guard let factor = factor, factor >= 1.0 else {
            throw ValidationError.invalidFactor("Linear RoPE requires factor >= 1.0")
        }
    }

    private func validateDynamicParameters() throws {
        guard let factor = factor, factor >= 1.0 else {
            throw ValidationError.invalidFactor("Dynamic RoPE requires factor >= 1.0")
        }
    }

    private func validateYarnParameters() throws {
        guard let factor = factor, factor >= 1.0 else {
            throw ValidationError.invalidFactor("Yarn RoPE requires factor >= 1.0")
        }

        if let attentionFactor = attentionFactor, attentionFactor <= 0 {
            throw ValidationError.invalidParameter("attention_factor must be > 0")
        }

        let betaFastValue = betaFast ?? 32.0
        let betaSlowValue = betaSlow ?? 1.0

        if betaFastValue < betaSlowValue {
            throw ValidationError.invalidParameter("beta_fast must be >= beta_slow")
        }
    }

    private func validateLongRopeParameters() throws {
        guard let shortFactor = shortFactor else {
            throw ValidationError.missingParameter("short_factor is required for longrope")
        }

        guard let longFactor = longFactor else {
            throw ValidationError.missingParameter("long_factor is required for longrope")
        }

        // Note: actual length validation would require config context
        // Here we just ensure they're non-empty
        if shortFactor.isEmpty {
            throw ValidationError.invalidParameter("short_factor cannot be empty")
        }

        if longFactor.isEmpty {
            throw ValidationError.invalidParameter("long_factor cannot be empty")
        }
    }

    private func validateLlama3Parameters() throws {
        guard let factor = factor, factor >= 1.0 else {
            throw ValidationError.invalidFactor("Llama3 RoPE requires factor >= 1.0")
        }

        guard originalMaxPositionEmbeddings != nil else {
            throw ValidationError.missingParameter("original_max_position_embeddings is required for llama3")
        }

        guard let lowFreqFactor = lowFreqFactor else {
            throw ValidationError.missingParameter("low_freq_factor is required for llama3")
        }

        guard let highFreqFactor = highFreqFactor else {
            throw ValidationError.missingParameter("high_freq_factor is required for llama3")
        }

        if highFreqFactor <= lowFreqFactor {
            throw ValidationError.invalidParameter("high_freq_factor must be > low_freq_factor")
        }
    }

    // MARK: - CustomStringConvertible

    var description: String {
        let dict = toDictionary()
        if let jsonData = try? JSONSerialization.data(withJSONObject: dict, options: .prettyPrinted),
           let jsonString = String(data: jsonData, encoding: .utf8)
        {
            return "RopeParameters(\(jsonString))"
        }
        return "RopeParameters(rope_theta: \(ropeTheta), rope_type: \(ropeType.rawValue))"
    }

    // MARK: - Equatable

    static func == (lhs: RopeParameters, rhs: RopeParameters) -> Bool {
        return lhs.ropeTheta == rhs.ropeTheta &&
            lhs.ropeType == rhs.ropeType &&
            lhs.factor == rhs.factor &&
            lhs.originalMaxPositionEmbeddings == rhs.originalMaxPositionEmbeddings &&
            lhs.attentionFactor == rhs.attentionFactor &&
            lhs.betaFast == rhs.betaFast &&
            lhs.betaSlow == rhs.betaSlow &&
            lhs.shortFactor == rhs.shortFactor &&
            lhs.longFactor == rhs.longFactor &&
            lhs.lowFreqFactor == rhs.lowFreqFactor &&
            lhs.highFreqFactor == rhs.highFreqFactor
    }
}

// MARK: - Validation Error

/// Error types for RoPE parameter validation.
enum ValidationError: Error, CustomStringConvertible {
    case invalidRopeType(String)
    case invalidFactor(String)
    case invalidParameter(String)
    case missingParameter(String)

    var description: String {
        switch self {
        case let .invalidRopeType(type):
            return "Invalid RoPE type: \(type). Must be one of: default, linear, dynamic, yarn, longrope, llama3"
        case let .invalidFactor(message):
            return "Invalid factor: \(message)"
        case let .invalidParameter(message):
            return "Invalid parameter: \(message)"
        case let .missingParameter(message):
            return "Missing required parameter: \(message)"
        }
    }
}

// MARK: - Convenience Extensions

extension RopeParameters {
    /// Creates default RoPE parameters with the specified base theta.
    ///
    /// - Parameter ropeTheta: The base period of the RoPE embeddings.
    /// - Returns: RopeParameters configured for default RoPE.
    static func `default`(ropeTheta: Double = 10000.0) -> RopeParameters {
        return RopeParameters(ropeTheta: ropeTheta, ropeType: .default)
    }

    /// Creates linear scaling RoPE parameters.
    ///
    /// - Parameters:
    ///   - ropeTheta: The base period of the RoPE embeddings.
    ///   - factor: The scaling factor (must be >= 1.0).
    /// - Returns: RopeParameters configured for linear RoPE.
    static func linear(ropeTheta: Double = 10000.0, factor: Double) -> RopeParameters {
        return RopeParameters(ropeTheta: ropeTheta, ropeType: .linear, factor: factor)
    }

    /// Creates dynamic NTK scaling RoPE parameters.
    ///
    /// - Parameters:
    ///   - ropeTheta: The base period of the RoPE embeddings.
    ///   - factor: The scaling factor (must be >= 1.0).
    /// - Returns: RopeParameters configured for dynamic RoPE.
    static func dynamic(ropeTheta: Double = 10000.0, factor: Double) -> RopeParameters {
        return RopeParameters(ropeTheta: ropeTheta, ropeType: .dynamic, factor: factor)
    }

    /// Creates Yarn RoPE parameters.
    ///
    /// - Parameters:
    ///   - ropeTheta: The base period of the RoPE embeddings.
    ///   - factor: The scaling factor (must be >= 1.0).
    ///   - attentionFactor: Optional attention scaling factor.
    ///   - betaFast: Extrapolation boundary (defaults to 32).
    ///   - betaSlow: Interpolation boundary (defaults to 1).
    /// - Returns: RopeParameters configured for Yarn RoPE.
    static func yarn(
        ropeTheta: Double = 10000.0,
        factor: Double,
        attentionFactor: Double? = nil,
        betaFast: Double? = 32.0,
        betaSlow: Double? = 1.0
    ) -> RopeParameters {
        return RopeParameters(
            ropeTheta: ropeTheta,
            ropeType: .yarn,
            factor: factor,
            attentionFactor: attentionFactor,
            betaFast: betaFast,
            betaSlow: betaSlow
        )
    }
}
