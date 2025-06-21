import { Request, Response, NextFunction } from 'express';
import { RateLimiterRedis } from 'rate-limiter-flexible';
import Redis from 'redis';

const redis = Redis.createClient({
  url: process.env.REDIS_URL || 'redis://localhost:6379'
});

const rateLimiter = new RateLimiterRedis({
  storeClient: redis,
  keyPrefix: 'oni_rlhf_rate_limit',
  points: 100, // Number of requests
  duration: 60, // Per 60 seconds
});

export const RateLimitMiddleware = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
  try {
    const key = req.ip || 'unknown';
    await rateLimiter.consume(key);
    next();
  } catch (rejRes: any) {
    const secs = Math.round(rejRes.msBeforeNext / 1000) || 1;
    res.set('Retry-After', String(secs));
    res.status(429).json({
      success: false,
      error: 'Too many requests',
      retryAfter: secs,
      timestamp: new Date()
    });
  }
};