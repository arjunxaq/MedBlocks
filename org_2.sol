// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract HospitalModelRegistry {
    // Struct to store both model and JSON stats
    struct ModelInfo {
        string modelHash;   // IPFS hash of .pth model
        string statsHash;   // IPFS hash of col_stats.json
    }

    // Mapping hospital → uploaded models
    mapping(address => ModelInfo[]) private modelData;
    address[] private registeredHospitals;

    event ModelUploaded(address indexed uploader, string modelHash, string statsHash);

    // Upload model (.pth) + stats (col_stats.json)
    function uploadModelData(string memory _modelHash, string memory _statsHash) public {
        if (modelData[msg.sender].length == 0) {
            registeredHospitals.push(msg.sender); // first-time uploader
        }
        modelData[msg.sender].push(ModelInfo({
            modelHash: _modelHash,
            statsHash: _statsHash
        }));

        emit ModelUploaded(msg.sender, _modelHash, _statsHash);
    }

    // Get all models for a hospital
    function getModelData(address _hospital) public view returns (ModelInfo[] memory) {
        return modelData[_hospital];
    }

    // Get the latest uploaded model info
    function getLatestModelData(address _hospital) public view returns (ModelInfo memory) {
        uint len = modelData[_hospital].length;
        require(len > 0, "No model uploaded by this hospital");
        return modelData[_hospital][len - 1];
    }

    // List all hospitals
    function getAllHospitals() public view returns (address[] memory) {
        return registeredHospitals;
    }
}