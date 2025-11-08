import Foundation

/// Configuration class to store the configuration of a NanoChatModel.
///
/// It is used to instantiate a NanoChat model according to the specified arguments,
/// defining the model architecture. Instantiating a configuration with the defaults
/// will yield a similar configuration to that of the karpathy/nanochat-d32.
///
/// Configuration objects inherit from `PreTrainedConfig` and can be used to control
/// the model outputs.
///
/// Example usage:
/// ```swift
/// // Initializing a NanoChat style configuration
/// let configuration = NanoChatConfig()
///
/// // Initializing a model from the NanoChat style configuration
/// let model = NanoChatModel(configuration: configuration)
///
/// // Accessing the model configuration
/// let config = model.config
/// ```
class NanoChatConfig: PreTrainedConfig {  
  /// Keys to ignore during inference when looking at model outputs.
  static let keysToIgnoreAtInference: [String] = ["past_key_values"]
  
  /// Tensor parallel plan for the base model, mapping layer names to parallelization strategies.
  static let baseModelTPPlan: [String: String] = [
    "layers.*.self_attn.q_proj": "colwise",
    "layers.*.self_attn.k_proj": "colwise",
    "layers.*.self_attn.v_proj": "colwise",
    "layers.*.self_attn.o_proj": "rowwise",
    "layers.*.mlp.fc1": "colwise",
    "layers.*.mlp.fc2": "rowwise"
  ]
  
  // MARK: - Instance Properties
  
  /// Vocabulary size of the NanoChat model.
  /// Defines the number of different tokens that can be represented by the inputs_ids.
  var vocabSize: Int
  
  /// Dimension of the hidden representations.
  var hiddenSize: Int
  
  /// Dimension of the MLP representations.
  /// If `nil`, it will be computed based on the model architecture.
  var intermediateSize: Int?
  
  /// Number of hidden layers in the Transformer decoder.
  var numHiddenLayers: Int
  
  /// Number of attention heads for each attention layer in the Transformer decoder.
  var numAttentionHeads: Int
  
  /// Number of key_value heads for Grouped Query Attention.
  /// - If `numKeyValueHeads == numAttentionHeads`, uses Multi Head Attention (MHA)
  /// - If `numKeyValueHeads == 1`, uses Multi Query Attention (MQA)
  /// - Otherwise uses Grouped Query Attention (GQA)
  var numKeyValueHeads: Int
  
  /// The maximum sequence length that this model might ever be used with.
  var maxPositionEmbeddings: Int
  
  /// The non-linear activation function in the decoder.
  var hiddenAct: String
  
  /// The dropout ratio for the attention probabilities.
  var attentionDropout: Double
  
  /// The epsilon used by the RMS normalization layers.
  var rmsNormEps: Double
  
  /// The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
  var initializerRange: Double
  
  /// Configuration parameters for the RoPE embeddings.
  /// Contains `rope_theta` and optionally parameters for scaling with longer `max_position_embeddings`.
  var ropeParams: RopeParameters?
  
  /// Whether or not the model should return the last key/values attentions.
  /// Only relevant if `config.isDecoder == true`.
  var useCache: Bool
  
  /// Scaling factor when applying tanh softcapping on the logits.
  var finalLogitSoftcapping: Double?
  
  /// Whether to use a bias in the query, key, and value projection layers during self-attention.
  var attentionBias: Bool
  
  // MARK: - Initialization
  
  /// Initialize a NanoChatConfig with the specified parameters.
  ///
  /// - Parameters:
  ///   - vocabSize: Vocabulary size of the model (default: 50304).
  ///   - hiddenSize: Dimension of hidden representations (default: 768).
  ///   - intermediateSize: Dimension of MLP representations (default: 8192).
  ///   - numHiddenLayers: Number of hidden layers (default: 12).
  ///   - numAttentionHeads: Number of attention heads (default: 6).
  ///   - numKeyValueHeads: Number of key-value heads (default: same as numAttentionHeads).
  ///   - maxPositionEmbeddings: Maximum sequence length (default: 2048).
  ///   - hiddenAct: Activation function (default: "relu2").
  ///   - attentionDropout: Attention dropout ratio (default: 0.0).
  ///   - rmsNormEps: RMS norm epsilon (default: 1e-6).
  ///   - initializerRange: Weight initialization std dev (default: 0.02).
  ///   - ropeParams: RoPE embedding parameters (optional).
  ///   - useCache: Whether to cache key/values (default: true).
  ///   - finalLogitSoftcapping: Logit softcapping scaling factor (default: 15.0).
  ///   - attentionBias: Whether to use bias in attention projections (default: false).
  ///   - bosTokenId: Beginning of stream token ID (default: 0).
  ///   - eosTokenId: End of stream token ID (default: 1).
  ///   - padTokenId: Padding token ID (default: 1).
  ///   - tieWordEmbeddings: Whether to tie word embeddings (default: false).
  ///   - additionalParams: Additional configuration parameters.
  init(
    vocabSize: Int = 50304,
    hiddenSize: Int = 768,
    intermediateSize: Int? = 8192,
    numHiddenLayers: Int = 12,
    numAttentionHeads: Int = 6,
    numKeyValueHeads: Int? = nil,
    maxPositionEmbeddings: Int = 2048,
    hiddenAct: String = "relu2",
    attentionDropout: Double = 0.0,
    rmsNormEps: Double = 1e-6,
    initializerRange: Double = 0.02,
    ropeParams: RopeParameters? = nil,
    useCache: Bool = true,
    finalLogitSoftcapping: Double? = 15.0,
    attentionBias: Bool = false,
    bosTokenId: Int = 0,
    eosTokenId: Int = 1,
    padTokenId: Int = 1,
    tieWordEmbeddings: Bool = false,
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
    self.ropeParams = ropeParams
    
    // Initialize base class
    super.init(
      outputHiddenStates: false,
      outputAttentions: false,
      returnDict: true,
      dtype: nil,
      tieWordEmbeddings: tieWordEmbeddings,
      chunkSizeFeedForward: 0,
      isEncoderDecoder: false,
      isDecoder: false,
      crossAttentionHiddenSize: nil,
      addCrossAttention: false,
      tieEncoderDecoder: false,
      architectures: nil,
      finetuningTask: nil,
      id2label: nil,
      label2id: nil,
      numLabels: nil,
      taskSpecificParams: nil,
      problemType: nil,
      tokenizerClass: nil,
      prefix: nil,
      bosTokenId: bosTokenId,
      padTokenId: padTokenId,
      eosTokenId: eosTokenId,
      sepTokenId: nil,
      decoderStartTokenId: nil,
      nameOrPath: "",
      additionalParams: additionalParams
    )
    
    // Validate and standardize RoPE parameters
    validateAndStandardizeRopeParameters()
  }
  
  /// Validates and standardizes the RoPE parameters.
  ///
  /// This ensures the RoPE configuration is correct and sets up default values if needed.
  private func validateAndStandardizeRopeParameters() {
    // If no rope_parameters provided, create default with rope_theta
    if ropeParams == nil {
      ropeParams = RopeParameters.default(ropeTheta: 10000.0)
    }
    
    // Validate the RoPE parameters
    if let ropeParameters = ropeParams {
      do {
        try ropeParameters.validate()
      } catch {
        ModelUtils.log("Warning: RoPE parameter validation failed: \(error)")
      }
    }
  }
}

// MARK: - Convenience Extensions

extension NanoChatConfig {
  /// Creates a NanoChatConfig from a dictionary.
  ///
  /// - Parameter dictionary: Dictionary containing configuration values.
  /// - Returns: A NanoChatConfig instance.
  static func fromDictionary(_ dictionary: [String: Any]) -> NanoChatConfig? {
    let vocabSize = dictionary["vocab_size"] as? Int ?? 50304
    let hiddenSize = dictionary["hidden_size"] as? Int ?? 768
    let intermediateSize = dictionary["intermediate_size"] as? Int
    let numHiddenLayers = dictionary["num_hidden_layers"] as? Int ?? 12
    let numAttentionHeads = dictionary["num_attention_heads"] as? Int ?? 6
    let numKeyValueHeads = dictionary["num_key_value_heads"] as? Int
    let maxPositionEmbeddings = dictionary["max_position_embeddings"] as? Int ?? 2048
    let hiddenAct = dictionary["hidden_act"] as? String ?? "relu2"
    let attentionDropout = dictionary["attention_dropout"] as? Double ?? 0.0
    let rmsNormEps = dictionary["rms_norm_eps"] as? Double ?? 1e-6
    let initializerRange = dictionary["initializer_range"] as? Double ?? 0.02
    let useCache = dictionary["use_cache"] as? Bool ?? true
    let finalLogitSoftcapping = dictionary["final_logit_softcapping"] as? Double
    let attentionBias = dictionary["attention_bias"] as? Bool ?? false
    let bosTokenId = dictionary["bos_token_id"] as? Int ?? 0
    let eosTokenId = dictionary["eos_token_id"] as? Int ?? 1
    let padTokenId = dictionary["pad_token_id"] as? Int ?? 1
    let tieWordEmbeddings = dictionary["tie_word_embeddings"] as? Bool ?? false
    
    // Handle rope_parameters
    var ropeParams: RopeParameters?
    if let ropeDict = dictionary["rope_parameters"] as? [String: Any] {
      ropeParams = RopeParameters.fromDictionary(ropeDict)
    }
    
    return NanoChatConfig(
      vocabSize: vocabSize,
      hiddenSize: hiddenSize,
      intermediateSize: intermediateSize,
      numHiddenLayers: numHiddenLayers,
      numAttentionHeads: numAttentionHeads,
      numKeyValueHeads: numKeyValueHeads,
      maxPositionEmbeddings: maxPositionEmbeddings,
      hiddenAct: hiddenAct,
      attentionDropout: attentionDropout,
      rmsNormEps: rmsNormEps,
      initializerRange: initializerRange,
      ropeParams: ropeParams,
      useCache: useCache,
      finalLogitSoftcapping: finalLogitSoftcapping,
      attentionBias: attentionBias,
      bosTokenId: bosTokenId,
      eosTokenId: eosTokenId,
      padTokenId: padTokenId,
      tieWordEmbeddings: tieWordEmbeddings
    )
  }
}
