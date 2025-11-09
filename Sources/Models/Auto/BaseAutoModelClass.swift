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
            // TO_DO: Call PreTrainedModel.fromPretrained() here
        }
        
        /*
        trust_remote_code = kwargs.get("trust_remote_code")
        kwargs["_from_auto"] = True
        hub_kwargs_names = [
            "cache_dir",
            "force_download",
            "local_files_only",
            "proxies",
            "revision",
            "subfolder",
            "token",
        ]
        hub_kwargs = {name: kwargs.pop(name) for name in hub_kwargs_names if name in kwargs}
        code_revision = kwargs.pop("code_revision", None)
        commit_hash = kwargs.pop("_commit_hash", None)
        adapter_kwargs = kwargs.pop("adapter_kwargs", None)

        token = hub_kwargs.pop("token", None)

        if token is not None:
            hub_kwargs["token"] = token
        */
        
    }
}
