import Foundation
import Hub
import Version

extension PreTrainedConfig {
    /// - Note: This is a deprecated way to get the config file, Hub API should be used instead
    private static func downloadUrl(_ url: String) async throws -> String {
        guard let downloadURL = URL(string: url) else {
            ModelUtils.log("Error: Invalid URL string: \(url)")
            throw AutoModelError.invalidConfig
        }

        // Create a temporary file URL with random filename
        let temporaryDirectoryURL = FileManager.default.temporaryDirectory
        let randomFileName = UUID().uuidString + ".json"
        let temporaryFileURL = temporaryDirectoryURL.appendingPathComponent(randomFileName)

        ModelUtils.log("Downloading from: \(url)")
        ModelUtils.log("Saving to temporary location: \(temporaryFileURL.path)")

        let (localURL, response) = try await URLSession.shared.download(from: downloadURL)

        // Validate HTTP response
        guard let httpResponse = response as? HTTPURLResponse,
              (200 ... 299).contains(httpResponse.statusCode)
        else {
            ModelUtils.log("Error: Invalid response type")
            throw AutoModelError.invalidConfig
        }

        do {
            // Move downloaded file to temporary location
            try FileManager.default.moveItem(at: localURL, to: temporaryFileURL)
            ModelUtils.log("Successfully downloaded file to: \(temporaryFileURL.path)")
            return temporaryFileURL.path
        } catch {
            ModelUtils.log("Error moving file: \(error.localizedDescription)")
            throw error
        }
    }

    private static func cachedFile(
        _ pretrainedModelNameOrPath: String,
        fileName: String,
        cacheDir _: String?,
        forceDownload _: Bool,
        localFilesOnly _: Bool,
        token _: String?,
        userAgent _: [String: Any]?,
        revision: String,
        subFolder _: String? = nil,
        commitHash _: String? = nil
    ) async throws -> String? {
        let downloadedRepoDir = await try HubApi.shared.snapshot(from: pretrainedModelNameOrPath, revision: revision, matching: [fileName])
        let filePath = downloadedRepoDir.appendingPathComponent(fileName)
        return FileManager.default.fileExists(atPath: filePath.path) ? filePath.path : nil
    }

    private static func parseConfigurationFile(
        _ pretrainedModelNameOrPath: String,
        configurationFileName: String? = nil,
        modelArguments: [String: Any]
    )
        async throws -> Config?
    {
        enum Constants {
            static let defaultConfigFile = "config.json"
            static let defaultRevision = "main"
        }

        let cacheDir = modelArguments["cache_dir"] as? String
        let forceDownload = modelArguments["force_download"] as? Bool ?? false
        let proxies = modelArguments["proxies"] as? [String: String]
        let token = modelArguments["token"] as? String
        let localFilesOnly = modelArguments["local_files_only"] as? Bool ?? false
        let revision = modelArguments["revision"] as? String ?? Constants.defaultRevision
        let subFolder = modelArguments["subfolder"] as? String
        let fromPipeline = modelArguments["_from_pipeline"] as? Bool
        let fromAutoClass = modelArguments["_from_auto"] as? Bool
        let ggufFile = modelArguments["gguf_file"] as? String

        var userAgent: [String: Any] = [:]
        userAgent["file_type"] = "config"
        userAgent["from_auto_class"] = fromAutoClass ?? false
        if let fromPipeline = fromPipeline {
            userAgent["using_pipeline"] = fromPipeline
        }

        // Check if pretrainedModelNameOrPath is a local directory
        var isDirectory: ObjCBool = false
        let fileManager = FileManager.default
        var isLocal = fileManager.fileExists(atPath: pretrainedModelNameOrPath, isDirectory: &isDirectory) && isDirectory.boolValue

        // Check if it's a direct file path
        let subfolder = subFolder ?? ""
        let filePath = (subfolder as NSString).appendingPathComponent(pretrainedModelNameOrPath)
        let isFile = fileManager.fileExists(atPath: filePath) && !isDirectory.boolValue

        var resolvedConfigFile: String?
        var configurationFile = configurationFileName
        if isFile {
            resolvedConfigFile = pretrainedModelNameOrPath
            isLocal = true
        } else if ModelUtils.isRemoteURL(pretrainedModelNameOrPath) {
            configurationFile = ggufFile != nil ? ggufFile : pretrainedModelNameOrPath
            resolvedConfigFile = try await downloadUrl(pretrainedModelNameOrPath)
        } else {
            configurationFile = ggufFile != nil ? ggufFile : (modelArguments["_configuration_file"] as? String ?? Constants.defaultConfigFile)
            resolvedConfigFile = await try cachedFile(
                pretrainedModelNameOrPath,
                fileName: configurationFile!,
                cacheDir: cacheDir,
                forceDownload: forceDownload,
                localFilesOnly: localFilesOnly,
                token: token,
                userAgent: userAgent,
                revision: revision
            )
        }

        if let ggufFile {
            ModelUtils.log("GGUF configuration loading is not right now supported!!")
            return nil
        }

        guard let resolvedConfigFile, let resolvedConfigFileUrl = URL(string: resolvedConfigFile) else {
            ModelUtils.log("Couldn't load the configuration file.")
            return nil
        }

        let configDict = try HubApi.shared.configuration(fileURL: resolvedConfigFileUrl)

        if isLocal {
            ModelUtils.log("Loading configuration file \(resolvedConfigFile)")
        } else {
            ModelUtils.log("Loading configuration file \(configurationFile) from cache at \(resolvedConfigFile)")
        }

        return configDict
    }

    static func getConfigurationFile(_ files: [String]) -> String? {
        enum Constants {
            static let defaultConfigFile = "config.json"
            static let configPrefix = "config."
            static let configSuffix = ".json"
        }

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
        var availableVersions =
            configurationFilesMap
                .keys
                .map { ($0, try? Version($0)) }
                .sorted {
                    if let leftVersion = $0.1, let rightVersion = $1.1 {
                        return leftVersion < rightVersion
                    } else if let leftVersion = $0.1 {
                        return true
                    } else {
                        return false
                    }
                }
                .map { $0.0 }

        // Defaults to the standard if can't parse version numbers properly
        var configurationFile = "config.json"

        // Get transformers version
        if let transformersVersion = try? Version(ModelUtils.version) {
            for version in availableVersions {
                if let configVersion = try? Version(version) {
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
        enum Constants {
            static let configurationFiles = "configuration_files"
        }

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
