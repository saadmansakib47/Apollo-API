import express from 'express';
import cors from 'cors';
import multer from 'multer';
import { createWorker } from 'tesseract.js';
import { Groq } from 'groq-sdk';
import dotenv from 'dotenv';
import rateLimit from 'express-rate-limit';
import mongoose from 'mongoose';
import { OAuth2Client } from 'google-auth-library';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 5000;

// Initialize Groq client
const groq = new Groq({
    apiKey: process.env.GROQ_API_KEY,
});

// Configure express trust proxy before rate limiter
app.set('trust proxy', 1);

// Rate limiting with custom key generator
const limiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 100, // limit each IP to 100 requests per windowMs
    standardHeaders: true,
    legacyHeaders: false,
    keyGenerator: (req) => req.ip || req.headers['x-forwarded-for'] || 'default-key',
});

// Middleware
app.use(cors({
    origin: 'http://localhost:5173', // Allow requests from Vite dev server
    credentials: true
}));
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));
app.use(limiter);

// Multer configuration for file uploads
const storage = multer.memoryStorage();
const upload = multer({
    storage: storage,
    limits: { fileSize: 5 * 1024 * 1024 }, // 5MB limit
    fileFilter: (req, file, cb) => {
        if (!file.mimetype.startsWith('image/')) {
            return cb(new Error('Only image files are allowed!'));
        }
        cb(null, true);
    }
});

// MongoDB connection
const connectDB = async () => {
    try {
        if (!process.env.MONGODB_URI) throw new Error('MONGODB_URI is not defined in environment variables');
        await mongoose.connect(process.env.MONGODB_URI, {
            serverSelectionTimeoutMS: 5000,
            socketTimeoutMS: 45000,
        });
        console.log('Connected to MongoDB');
    } catch (err) {
        console.error('MongoDB connection error:', err);
        console.error('Continuing without database connection...');
    }
};
connectDB();

// Authentication Middleware (Google ID Token Verification)
const client = new OAuth2Client(process.env.GOOGLE_CLIENT_ID);

const verifyGoogleToken = async (req, res, next) => {
    try {
        const token = req.headers['authorization']?.split(' ')[1]; // Extract Bearer token
        if (!token) return res.status(401).json({ message: 'Unauthorized: No token provided' });

        const ticket = await client.verifyIdToken({
            idToken: token,
            audience: process.env.GOOGLE_CLIENT_ID, // Must match frontend Client ID
        });

        const payload = ticket.getPayload();
        req.user = {
            userId: payload.sub,  // Google User ID
            email: payload.email,
            name: payload.name,
            picture: payload.picture
        };

        next(); // Proceed to next middleware
    } catch (error) {
        console.error('Google Token Verification Failed:', error);
        return res.status(403).json({ message: 'Invalid or expired token' });
    }
};

// Report Schema
const reportSchema = new mongoose.Schema({
    userId: String,  // Link reports to authenticated users
    originalText: String,
    summary: String,
    analysis: {
        keyFindings: [String],
        recommendations: [String],
        urgentConcerns: [String],
        simplifiedExplanation: String
    },
    createdAt: { type: Date, default: Date.now }
});

const Report = mongoose.model('Report', reportSchema);

// Process report endpoint (Requires Authentication)
app.post('/api/analyze-report', verifyGoogleToken, upload.single('report'), async (req, res) => {
    try {
        if (!req.file) return res.status(400).json({ error: 'No file uploaded' });

        // Initialize Tesseract worker
        const worker = await createWorker();
        const { data: { text } } = await worker.recognize(req.file.buffer);
        await worker.terminate();

        // Generate analysis using Groq
        const completion = await groq.chat.completions.create({
            messages: [
                { role: "system", content: "You are a medical expert AI assistant specialized in analyzing medical reports and explaining them in simple terms." },
                { role: "user", content: `Analyze this report:\n\n${text}` }
            ],
            model: "mixtral-8x7b-32768",
            temperature: 0.5,
            max_tokens: 1024,
        });

        const analysis = completion.choices[0]?.message?.content;

        // Parse structured response
        const sections = analysis.split('\n\n');
        const parsedAnalysis = {
            keyFindings: sections[0]?.split('\n').filter(line => line.startsWith('-')).map(line => line.substring(2)),
            recommendations: sections[1]?.split('\n').filter(line => line.startsWith('-')).map(line => line.substring(2)),
            urgentConcerns: sections[2]?.split('\n').filter(line => line.startsWith('-')).map(line => line.substring(2)),
            simplifiedExplanation: sections[3]
        };

        // Save to database
        try {
            const report = new Report({
                userId: req.user.userId, // Store user ID
                originalText: text,
                summary: analysis,
                analysis: parsedAnalysis
            });
            await report.save();
        } catch (dbError) {
            console.error('Database save error:', dbError);
        }

        res.json({
            success: true,
            text,
            analysis: parsedAnalysis
        });

    } catch (error) {
        console.error('Error processing report:', error);
        res.status(500).json({ error: 'Error processing report', details: error.message });
    }
});

// Get report history (Requires Authentication)
app.get('/api/reports', verifyGoogleToken, async (req, res) => {
    try {
        const reports = await Report.find({ userId: req.user.userId }).sort({ createdAt: -1 }).limit(10);
        res.json(reports);
    } catch (error) {
        res.status(500).json({ error: 'Error fetching reports' });
    }
});

// Error handling middleware
app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(500).json({ error: 'Something went wrong!', details: err.message });
});

app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});
