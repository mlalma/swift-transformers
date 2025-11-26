import Foundation
import Hub

public class BaseAutoModelClass {            
    internal init() {
        // This class can't be directly instantiated, use one of the derived classes to initialize it
    }

    internal func from(
        pretrained pretrainedModelNameOrPath: String,
        modelArguments: [String: Any] = [:],
        modelMapping: [String: (PreTrainedConfig) throws -> PreTrainedModel])
    async -> PreTrainedModel? {
        let useSafetensors = modelArguments["use_safetensors"] as? Bool
        var modelConfig = modelArguments["config"] as? PreTrainedConfig

        if modelConfig == nil {
            modelConfig = await AutoConfig.from(pretrained: pretrainedModelNameOrPath, modelArguments: modelArguments)
        }
        
        guard let modelConfig,
              let modelType = modelConfig.modelType else {
            ModelUtils.log("Could not load config file from \(pretrainedModelNameOrPath)")
            return nil
        }
        
        guard let model = try? modelMapping[modelType]?(modelConfig) else {
            ModelUtils.log("Could not instantiate model of type \(modelType)")
            return nil
        }
        
        do {
            try await model.fromPretrained(pretrainedModelNameOrPath, config: modelConfig, useSafetensors: useSafetensors, modelArguments: modelArguments)
            return model
        } catch {
            ModelUtils.log("Error when initializing model \(error)")
            return nil
        }
    }
}
