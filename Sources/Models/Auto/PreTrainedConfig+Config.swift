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

    private static func parseConfigurationFile(
        _ pretrainedModelNameOrPath: String,
        configurationFileName: String? = nil,
        modelArguments: [String: Any]
    )
        async throws -> Config?
    {
        let cacheDir = modelArguments[Constants.cacheDir] as? String
        let forceDownload = modelArguments[Constants.forceDownload] as? Bool ?? false
        let proxies = modelArguments[Constants.proxies] as? [String: String]
        let token = modelArguments[Constants.token] as? String
        let localFilesOnly = modelArguments[Constants.localFilesOnly] as? Bool ?? false
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
            configurationFile = ggufFile != nil ? ggufFile : (configurationFile != nil ? configurationFile! : Constants.defaultConfigFile)
            guard let configurationFile else { return nil }
            let downloadedRepoDir = await try HubApi.shared.snapshot(from: pretrainedModelNameOrPath, revision: revision, matching: [configurationFile])
            let filePath = downloadedRepoDir.appendingPathComponent(configurationFile)
            resolvedConfigFile = FileManager.default.fileExists(atPath: filePath.path) ? filePath.path : nil
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
            ModelUtils.log("Did load configuration file \(resolvedConfigFile)")
        } else {
            ModelUtils.log("Did load configuration file \(configurationFile) from cache at \(resolvedConfigFile)")
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
        var configurationFile = Constants.defaultConfigFile

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

extension PreTrainedConfig.Constants {
    static let defaultConfigFile = "config.json"
    static let defaultRevision = "main"

    static let configPrefix = "config."
    static let configSuffix = ".json"

    // Model argument keys
    static let cacheDir = "cache_dir"
    static let forceDownload = "force_download"
    static let proxies = "proxies"
    static let token = "token"
    static let localFilesOnly = "local_files_only"
    static let revision = "revision"
    static let subfolder = "subfolder"
    static let fromPipeline = "_from_pipeline"
    static let fromAuto = "_from_auto"
    static let ggufFile = "gguf_file"

    // User agent keys
    static let fileType = "file_type"
    static let fromAutoClass = "from_auto_class"
    static let usingPipeline = "using_pipeline"
}
