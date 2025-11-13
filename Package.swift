// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "swift-transformers",
    platforms: [.iOS(.v18), .macOS(.v15)],
    products: [
        .library(name: "Hub", targets: ["Hub"]),
        .library(name: "Tokenizers", targets: ["Tokenizers"]),
        .library(name: "Transformers", targets: ["Tokenizers", "Generation", "Models"]),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-jinja.git", from: "2.0.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.29.1"),
        .package(url: "https://github.com/mxcl/Version.git", from: "2.0.0"),
        .package(url: "https://github.com/mlalma/PTReaderSwift.git", from: "0.0.4"),
    ],
    targets: [
        .target(
            name: "Generation",
            dependencies: ["Tokenizers"]
        ),
        .target(
            name: "Hub",
            dependencies: [.product(name: "Jinja", package: "swift-jinja")],
            resources: [.process("Resources")]
        ),
        .target(
            name: "Models",
            dependencies: [
                "Tokenizers",
                "Generation",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "Version", package: "Version"),
                .product(name: "PTReaderSwift", package: "PTReaderSwift"),
            ]
        ),
        .target(
            name: "Tokenizers",
            dependencies: ["Hub", .product(name: "Jinja", package: "swift-jinja")]
        ),
        .testTarget(
            name: "GenerationTests",
            dependencies: ["Generation"]
        ),
        .testTarget(
            name: "HubTests",
            dependencies: ["Hub", .product(name: "Jinja", package: "swift-jinja")]
        ),
        .testTarget(
            name: "ModelsTests",
            dependencies: ["Models", "Hub"],
            resources: [.process("Resources")]
        ),
        .testTarget(
            name: "TokenizersTests",
            dependencies: ["Tokenizers", "Models", "Hub"],
            resources: [.process("Resources")]
        ),
    ]
)
