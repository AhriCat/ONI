import React from 'react';
import ChatContainer from './components/Chat/ChatContainer';
import Sidebar from './components/Sidebar/Sidebar';
import WalletModal from './components/Wallet/WalletModal';
import Notification from './components/UI/Notification';
import { WalletProvider } from './contexts/WalletContext';
import { RLHFProvider } from './contexts/RLHFContext';
import { ChatProvider } from './contexts/ChatContext';
import './styles/globals.css';

const App: React.FC = () => {
  return (
    <WalletProvider>
      <RLHFProvider>
        <ChatProvider>
          <div className="chat-container">
            <Sidebar />
            <ChatContainer />
            <WalletModal />
            <Notification />
          </div>
        </ChatProvider>
      </RLHFProvider>
    </WalletProvider>
  );
};

export default App;
