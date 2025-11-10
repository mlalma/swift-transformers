import Foundation
import Hub
import Version

extension PreTrainedConfig {
    private static func parseConfigurationFile(
        _ pretrainedModelNameOrPath: String,
        configurationFileName: String? = nil,
        modelArguments: [String: Any]
    )
        async throws -> Config?
    {
        // let cacheDir = modelArguments[Constants.cacheDir] as? String
        // let forceDownload = modelArguments[Constants.forceDownload] as? Bool ?? false
        // let proxies = modelArguments[Constants.proxies] as? [String: String]
        // let token = modelArguments[Constants.token] as? String
        // let localFilesOnly = modelArguments[Constants.localFilesOnly] as? Bool ?? false
        let revision = modelArguments[Constants.revision] as? String ?? Constants.defaultRevision
        let subFolder = modelArguments[Constants.subfolder] as? String
        let fromPipeline = modelArguments[Constants.fromPipeline] as? Bool
        let fromAutoClass = modelArguments[Constants.fromAuto] as? Bool
        let ggufFile = modelArguments[Constants.ggufFile] as? String

        var userAgent: [String: Any] = [:]
        userAgent[Constants.fileType] = "config"
        userAgent[Constants.fromAutoClass] = fromAutoClass ?? false
        if let fromPipeline = fromPipeline {
            userAgent[Constants.usingPipeline] = fromPipeline
        }

        let localDirOrFile = ModelUtils.isLocal(pretrainedModelNameOrPath, subFolder)

        var resolvedConfigFile: String?
        var configurationFile = configurationFileName
        if localDirOrFile == .file || localDirOrFile == .directory {
            resolvedConfigFile = pretrainedModelNameOrPath
        } else if ModelUtils.isRemote(pretrainedModelNameOrPath) {
            configurationFile = ggufFile != nil ? ggufFile : pretrainedModelNameOrPath
            resolvedConfigFile = try await ModelUtils.downloadUrl(pretrainedModelNameOrPath)
        } else {
            configurationFile = ggufFile != nil ? ggufFile : (configurationFile != nil ? configurationFile! : Constants.defaultConfigFile)
            guard let configurationFile else { return nil }
            let downloadedRepoDir = try await HubApi.shared.snapshot(from: pretrainedModelNameOrPath, revision: revision, matching: [configurationFile])
            let filePath = downloadedRepoDir.appendingPathComponent(configurationFile)
            resolvedConfigFile = FileManager.default.fileExists(atPath: filePath.path) ? filePath.path : nil
        }

        if ggufFile != nil {
            ModelUtils.log("GGUF configuration loading is not right now supported!!")
            return nil
        }

        guard let resolvedConfigFile else {
            ModelUtils.log("Couldn't load the configuration file.")
            return nil
        }

        let resolvedConfigFileUrl = URL(fileURLWithPath: resolvedConfigFile)
        let configDict = try HubApi.shared.configuration(fileURL: resolvedConfigFileUrl)

        if localDirOrFile != .notLocal {
            ModelUtils.log("Did load configuration file \(resolvedConfigFile)")
        } else {
            ModelUtils.log("Did load configuration file \(configurationFile ?? "NO_FILE") from cache at \(resolvedConfigFile)")
        }

        return configDict
    }

    static func getConfigurationFile(_ files: [String]) -> String? {
        var configurationFilesMap: [String: String] = [:]

        for fileName in files {
            if fileName.hasPrefix(Constants.configPrefix),
               fileName.hasSuffix(Constants.configSuffix),
               fileName != Constants.defaultConfigFile
            {
                // Remove "config." prefix and ".json" suffix to get version
                var version = fileName
                if version.hasPrefix(Constants.configPrefix) {
                    version = String(version.dropFirst(Constants.configPrefix.count))
                }
                if version.hasSuffix(Constants.configSuffix) {
                    version = String(version.dropLast(Constants.configSuffix.count))
                }
                configurationFilesMap[version] = fileName
            }
        }

        // Perform semantic sort
        let availableVersions =
            configurationFilesMap
                .keys
                .map { ($0, Version($0)) }
                .sorted {
                    if let leftVersion = $0.1, let rightVersion = $1.1 {
                        return leftVersion < rightVersion
                    } else if let _ = $0.1 {
                        return true
                    } else {
                        return false
                    }
                }
                .map { $0.0 }

        // Defaults to the standard if can't parse version numbers properly
        var configurationFile = Constants.defaultConfigFile

        // Get transformers version
        if let transformersVersion = Version(ModelUtils.version) {
            for version in availableVersions {
                if let configVersion = Version(version) {
                    if configVersion <= transformersVersion,
                       let versionConfigFile = configurationFilesMap[version]
                    {
                        configurationFile = versionConfigFile
                    } else {
                        // No point going further since the versions are sorted.
                        break
                    }
                }
            }
        }

        return configurationFile
    }

    static func getConfigDict(_ pretrainedModelNameOrPath: String, modelArguments: [String: Any]) async -> Config? {
        var configDict: Config?

        do {
            configDict = try await parseConfigurationFile(pretrainedModelNameOrPath, modelArguments: modelArguments)

            if let configurationFileList = configDict?[Constants.configurationFiles, [String].self],
               let configurationFile = PreTrainedConfig.getConfigurationFile(configurationFileList)
            {
                configDict = try await parseConfigurationFile(
                    pretrainedModelNameOrPath,
                    configurationFileName: configurationFile,
                    modelArguments: modelArguments
                )
            }
        } catch {
            ModelUtils.log("Exception: \(error)")
            configDict = nil
        }

        return configDict
    }
}

extension PreTrainedConfig.Constants {
    static let defaultConfigFile = "config.json"
    static let defaultRevision = "main"

    static let configPrefix = "config."
    static let configSuffix = ".json"

    // Model argument keys
    static let cacheDir = "cache_dir"
    static let configurationFiles = "configuration_files"
    static let forceDownload = "force_download"
    static let fromAuto = "_from_auto"
    static let fromPipeline = "_from_pipeline"
    static let ggufFile = "gguf_file"
    static let localFilesOnly = "local_files_only"
    static let proxies = "proxies"
    static let token = "token"
    static let revision = "revision"
    static let subfolder = "subfolder"

    // User agent keys
    static let fileType = "file_type"
    static let fromAutoClass = "from_auto_class"
    static let usingPipeline = "using_pipeline"
}
