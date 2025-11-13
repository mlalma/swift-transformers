import Foundation

struct DownloadArguments {
    let cacheDir: String?
    let forceDownload: Bool
    let proxies: [String: String]?
    let localFilesOnly: Bool
    let token: String?
    let revision: String?
    let subFolder: String
    let commitHash: String?
}
