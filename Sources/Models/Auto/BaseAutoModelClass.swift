import Foundation
import Hub

public class BaseAutoModelClass {
        
    
    init() {
        // This class can't be directly instantiated, use one of the derived classes to initialize it
    }

    internal func fromPretrained(_ pretrainedModelNameOrPath: String, modelArguments: [String: Any] = [:], modelMapping: [String: () -> PreTrainedModel]) async -> PreTrainedModel? {
        let useSafetensors = modelArguments["use_safetensors"] as? Bool
        var modelConfig = modelArguments["config"] as? PreTrainedConfig

        if modelConfig == nil {
            modelConfig = await AutoConfig.fromPretrained(pretrainedModelNameOrPath, modelArguments: modelArguments)
        }
        
        guard let modelConfig,
              let modelType = modelConfig.modelType else {
            ModelUtils.log("Could not load config file from \(pretrainedModelNameOrPath)")
            return nil
        }
        
        guard let model = modelMapping[modelType]?() else {
            ModelUtils.log("Could not instantiate model of type \(modelType)")
            return nil
        }
        
        do {
            return try await model.fromPretrained(pretrainedModelNameOrPath, config: modelConfig, useSafetensors: useSafetensors, modelArguments: modelArguments)
        } catch {
            ModelUtils.log("Error when initializing model \(error)")
            return nil
        }
    }
}
