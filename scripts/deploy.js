const { ethers } = require("hardhat");

async function main() {
  console.log("Deploying ONI Token contract...");

  // Get the contract factory
  const ONIToken = await ethers.getContractFactory("ONIToken");

  // Deploy the contract
  const oniToken = await ONIToken.deploy();

  // Wait for deployment to complete
  await oniToken.waitForDeployment();

  // Get the contract address
  const contractAddress = await oniToken.getAddress();
  console.log("ONI Token deployed to:", contractAddress);

  // Verify contract on block explorer if not on a local network
  const networkName = hre.network.name;
  if (networkName !== "hardhat" && networkName !== "localhost") {
    console.log("Waiting for block confirmations...");
    
    // Wait for 6 block confirmations
    await oniToken.deploymentTransaction().wait(6);
    
    console.log("Verifying contract on block explorer...");
    try {
      await hre.run("verify:verify", {
        address: contractAddress,
        constructorArguments: [],
      });
      console.log("Contract verified successfully");
    } catch (error) {
      console.error("Error verifying contract:", error);
    }
  }

  // Log deployment info
  console.log("\nDeployment Summary:");
  console.log("--------------------");
  console.log("Contract: ONIToken");
  console.log("Address:", contractAddress);
  console.log("Network:", networkName);
  console.log("Deployer:", (await ethers.getSigners())[0].address);
  console.log("Block:", await ethers.provider.getBlockNumber());
  console.log("Timestamp:", Math.floor(Date.now() / 1000));
  console.log("Gas Price:", ethers.formatUnits(await ethers.provider.getFeeData().then(data => data.gasPrice), "gwei"), "gwei");
  console.log("--------------------");
  
  // Save deployment info to a file
  const fs = require("fs");
  const deploymentInfo = {
    contract: "ONIToken",
    address: contractAddress,
    network: networkName,
    deployer: (await ethers.getSigners())[0].address,
    block: await ethers.provider.getBlockNumber(),
    timestamp: Math.floor(Date.now() / 1000),
    gasPrice: ethers.formatUnits(await ethers.provider.getFeeData().then(data => data.gasPrice), "gwei")
  };
  
  fs.writeFileSync(
    `deployment-${networkName}-${Math.floor(Date.now() / 1000)}.json`,
    JSON.stringify(deploymentInfo, null, 2)
  );
  
  console.log("Deployment info saved to file");
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });