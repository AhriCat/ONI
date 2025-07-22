require("@nomicfoundation/hardhat-toolbox");
require("dotenv").config();

/** @type import('hardhat/config').HardhatUserConfig */
module.exports = {
  solidity: {
    version: "0.8.19",
    settings: {
      optimizer: {
        enabled: true,
        runs: 200
      }
    }
  },
  networks: {
    hardhat: {
      chainId: 1337
    },
    oni: {
      url: process.env.ONI_RPC_URL || "http://localhost:8545",
      accounts: process.env.PRIVATE_KEY ? [process.env.PRIVATE_KEY] : [],
      chainId: parseInt(process.env.ONI_CHAIN_ID || "31337"),
      gasPrice: parseInt(process.env.ONI_GAS_PRICE || "1000000000") // 1 gwei
    },
    oni_testnet: {
      url: process.env.ONI_TESTNET_RPC_URL || "http://testnet.oni-chain.com:8545",
      accounts: process.env.PRIVATE_KEY ? [process.env.PRIVATE_KEY] : [],
      chainId: parseInt(process.env.ONI_TESTNET_CHAIN_ID || "31338"),
      gasPrice: parseInt(process.env.ONI_TESTNET_GAS_PRICE || "500000000") // 0.5 gwei
    }
  },
  paths: {
    sources: "./contracts",
    tests: "./test",
    cache: "./cache",
    artifacts: "./artifacts"
  },
  mocha: {
    timeout: 40000
  },
  etherscan: {
    apiKey: process.env.ETHERSCAN_API_KEY
  }
};
