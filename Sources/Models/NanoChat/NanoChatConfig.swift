import Foundation
import Hub

/// Configuration class to store the configuration of a NanoChatModel.
final class NanoChatConfig: PreTrainedConfig {
    /// Keys to ignore during inference when looking at model outputs.
    static let keysToIgnoreAtInference: [String] = ["past_key_values"]

    /// Tensor parallel plan for the base model, mapping layer names to parallelization strategies.
    static let baseModelTPPlan: [String: String] = [
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.fc1": "colwise",
        "layers.*.mlp.fc2": "rowwise",
    ]

    var attentionBias: Bool
    var attentionDropout: Double
    var hiddenAct: String
    var hiddenSize: Int
    var initializerRange: Double
    var intermediateSize: Int
    var finalLogitSoftcapping: Double?
    var maxPositionEmbeddings: Int
    var numAttentionHeads: Int
    var numHiddenLayers: Int
    var numKeyValueHeads: Int
    var rmsNormEps: Double
    var useCache: Bool
    var vocabSize: Int

    /// Initialize a NanoChatConfig with the specified parameters.
    init(
        attentionBias: Bool = Constants.attentionBias,
        attentionDropout: Double = Constants.attentionDropout,
        bosTokenId: Int = Constants.bosTokenId,
        dtype: String = Constants.dtype,
        eosTokenId: Int = Constants.eosTokenId,
        hiddenAct: String = Constants.hiddenAct,
        hiddenSize: Int = Constants.hiddenSize,
        initializerRange: Double = Constants.initializerRange,
        intermediateSize: Int = Constants.intermediateSize,
        finalLogitSoftcapping: Double? = Constants.finalLogitSoftcapping,
        maxPositionEmbeddings: Int = Constants.maxPositionEmbeddings,
        numAttentionHeads: Int = Constants.numAttentionHeads,
        numHiddenLayers: Int = Constants.numHiddenLayers,
        numKeyValueHeads: Int? = Constants.numKeyValueHeads,
        padTokenId: Int = Constants.padTokenId,
        rmsNormEps: Double = Constants.rmsNormEps,
        ropeParameters: RopeParameters = RopeParameters.default(ropeTheta: Constants.ropeTheta),
        tieWordEmbeddings: Bool = Constants.tieWordEmbeddings,
        useCache: Bool = Constants.useCache,
        vocabSize: Int = Constants.vocabSize,
        additionalParams: [String: Any] = [:]
    ) {
        // Initialize model-specific properties
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads

        // For backward compatibility: if numKeyValueHeads is nil, default to numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads ?? numAttentionHeads

        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.hiddenAct = hiddenAct
        self.attentionDropout = attentionDropout
        self.rmsNormEps = rmsNormEps
        self.initializerRange = initializerRange
        self.useCache = useCache
        self.finalLogitSoftcapping = finalLogitSoftcapping
        self.attentionBias = attentionBias

        // Initialize base class
        super.init(
            dtype: dtype,
            tieWordEmbeddings: tieWordEmbeddings,
            bosTokenId: bosTokenId,
            padTokenId: padTokenId,
            eosTokenId: eosTokenId,
            ropeParameters: ropeParameters,
            additionalParams: additionalParams
        )

        modelType = "nanochat"

        // Validate and standardize RoPE parameters
        validateAndStandardizeRopeParameters()
    }

    /// Initialize from parsed configuration class
    override init(fromConfig config: Config) {
        // Parse NanoChat-specific configuration values
        attentionBias = config[ConfigKeys.attentionBias, Bool.self] ?? Constants.attentionBias
        attentionDropout = config[ConfigKeys.attentionDropout, Double.self] ?? Constants.attentionDropout
        hiddenAct = config[ConfigKeys.hiddenAct, String.self] ?? Constants.hiddenAct
        hiddenSize = config[ConfigKeys.hiddenSize, Int.self] ?? Constants.hiddenSize
        initializerRange = config[ConfigKeys.initializerRange, Double.self] ?? Constants.initializerRange
        intermediateSize = config[ConfigKeys.intermediateSize, Int.self] ?? Constants.intermediateSize
        finalLogitSoftcapping = config[ConfigKeys.logitsSoftCap, Double.self] ?? Constants.finalLogitSoftcapping
        maxPositionEmbeddings = config[ConfigKeys.maxPositionEmbeddings, Int.self] ?? Constants.maxPositionEmbeddings
        numAttentionHeads = config[ConfigKeys.numAttentionHeads, Int.self] ?? Constants.numAttentionHeads
        numHiddenLayers = config[ConfigKeys.numHiddenLayers, Int.self] ?? Constants.numHiddenLayers
        numKeyValueHeads = config[ConfigKeys.numKeyValueHeads, Int.self] ?? config[ConfigKeys.numAttentionHeads, Int.self] ?? Constants.numKeyValueHeads
        rmsNormEps = config[ConfigKeys.rmsNormEps, Double.self] ?? Constants.rmsNormEps
        useCache = config[ConfigKeys.useCache, Bool.self] ?? Constants.useCache
        vocabSize = config[ConfigKeys.vocabSize, Int.self] ?? Constants.vocabSize

        // Call super.init to initialize base class properties
        super.init(fromConfig: config)

        // Validate and standardize RoPE parameters
        validateAndStandardizeRopeParameters()
    }

    /// Validates and standardizes the RoPE parameters.
    private func validateAndStandardizeRopeParameters() {
        // If no rope_parameters provided, create default with rope_theta
        if ropeParameters == nil {
            ropeParameters = RopeParameters.default(ropeTheta: Constants.ropeTheta)
        }

        // Validate the RoPE parameters
        if let ropeParameters {
            do {
                try ropeParameters.validate()
            } catch {
                ModelUtils.log("Warning: RoPE parameter validation failed: \(error)")
            }
        }
    }

    /// Default values for NanoChat configuration parameters
    enum Constants {
        // Model architecture
        static let attentionBias = false
        static let attentionDropout = 0.0
        static let hiddenAct = "relu2"
        static let hiddenSize = 2048
        static let initializerRange = 0.02
        static let intermediateSize = 8192
        static let finalLogitSoftcapping = 15.0
        static let maxPositionEmbeddings = 2048
        static let numAttentionHeads = 16
        static let numHiddenLayers = 32
        static let numKeyValueHeads = 16
        static let rmsNormEps = 1e-6
        static let useCache = true
        static let vocabSize = 65536

        // Token IDs
        static let bosTokenId = 65527
        static let eosTokenId = 65531
        static let padTokenId = 65531

        // Model settings
        static let dtype = "bfloat16"
        static let tieWordEmbeddings = false

        // RoPE settings
        static let ropeTheta = 10000.0
    }

    /// Config key names (snake_case strings used in configuration files)
    enum ConfigKeys {
        static let attentionBias = "attention_bias"
        static let attentionDropout = "attention_dropout"
        static let hiddenAct = "hidden_act"
        static let hiddenSize = "hidden_size"
        static let initializerRange = "initializer_range"
        static let intermediateSize = "intermediate_size"
        static let logitsSoftCap = "logits_soft_cap"
        static let maxPositionEmbeddings = "max_position_embeddings"
        static let numAttentionHeads = "num_attention_heads"
        static let numHiddenLayers = "num_hidden_layers"
        static let numKeyValueHeads = "num_key_value_heads"
        static let rmsNormEps = "rms_norm_eps"
        static let useCache = "use_cache"
        static let vocabSize = "vocab_size"
    }
}
