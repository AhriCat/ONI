import { Request, Response, NextFunction } from 'express';
import jwt from 'jsonwebtoken';
import { User } from '../types';

export const AuthMiddleware = (req: Request, res: Response, next: NextFunction): void => {
  const token = req.headers.authorization?.replace('Bearer ', '');
  
  if (!token) {
    res.status(401).json({
      success: false,
      error: 'Authentication token required',
      timestamp: new Date()
    });
    return;
  }

  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET || 'default-secret') as any;
    
    // Mock user for demonstration - in production, fetch from database
    const user: User = {
      id: decoded.userId || 'user-123',
      username: decoded.username || 'demo-user',
      email: decoded.email || 'demo@example.com',
      role: decoded.role || 'contributor',
      reputation: decoded.reputation || 100,
      totalContributions: decoded.totalContributions || 0,
      verificationLevel: decoded.verificationLevel || 'email',
      walletAddress: decoded.walletAddress,
      apiKey: decoded.apiKey || 'demo-api-key',
      permissions: decoded.permissions || ['download_models', 'submit_feedback', 'train_models'],
      createdAt: new Date(decoded.createdAt || Date.now()),
      lastActive: new Date()
    };

    (req as any).user = user;
    next();
  } catch (error) {
    res.status(401).json({
      success: false,
      error: 'Invalid authentication token',
      timestamp: new Date()
    });
  }
};