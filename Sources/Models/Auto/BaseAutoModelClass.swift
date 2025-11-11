import Foundation
import Hub

public class BaseAutoModelClass {
    init() {
        // This class can't be directly instantiated, use one of the derived classes to initialize it
    }

    func fromPretrained(_ pretrainedModelNameOrPath: String, modelArguments: [String: Any] = [:]) async {
        var kwargs = modelArguments
        var modelConfig = kwargs["config"] as? PreTrainedConfig

        let hubArgumentName = [
            "cache_dir",
            "force_download",
            "local_files_only",
            "proxies",
            "revision",
            "subfolder",
            "token",
        ]

        var hubArgs: [String: Any] = [:]
        for name in hubArgumentName {
            if let value = kwargs[name] {
                hubArgs[name] = value
                kwargs.removeValue(forKey: name)
            }
        }

        if modelConfig == nil {
            modelConfig = await AutoConfig.fromPretrained(pretrainedModelNameOrPath, modelArguments: modelArguments)
            
            if modelConfig == nil {
                ModelUtils.log("Could not load config file from \(pretrainedModelNameOrPath)")
                return
            }
        }
        
        
    }
}
