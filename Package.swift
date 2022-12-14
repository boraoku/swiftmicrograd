// swift-tools-version: 5.6
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "SwiftMicroGrad",
    products: [
        .library(
            name: "SwiftMicroGrad",
            targets: ["SwiftMicroGrad"]),
    ],
    dependencies: [],
    targets: [
        .target(
            name: "SwiftMicroGrad",
            dependencies: []),
        .testTarget(
            name: "SwiftMicroGradTests",
            dependencies: ["SwiftMicroGrad"]),
    ]
)
