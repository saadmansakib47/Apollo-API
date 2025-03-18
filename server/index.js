import express from 'express';
import cors from 'cors';
import multer from 'multer';
import { createWorker } from 'tesseract.js';
import { Groq } from 'groq-sdk';
import dotenv from 'dotenv';
import rateLimit from 'express-rate-limit';
import mongoose from 'mongoose';

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
    keyGenerator: (req) => {
        return req.ip || req.headers['x-forwarded-for'] || 'default-key';
    }
});

// Middleware
app.use(cors({
    origin: 'http://localhost:3000', // Allow requests from Vite dev server
    credentials: true
}));
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));
app.use(limiter);

// Multer configuration for file uploads
const storage = multer.memoryStorage();
const upload = multer({
    storage: storage,
    limits: {
        fileSize: 5 * 1024 * 1024 // 5MB limit
    },
    fileFilter: (req, file, cb) => {
        if (!file.mimetype.startsWith('image/')) {
            return cb(new Error('Only image files are allowed!'));
        }
        cb(null, true);
    }
});

// MongoDB connection with proper error handling
const connectDB = async () => {
    try {
        if (!process.env.MONGODB_URI) {
            throw new Error('MONGODB_URI is not defined in environment variables');
        }

        await mongoose.connect(process.env.MONGODB_URI, {
            serverSelectionTimeoutMS: 5000,
            socketTimeoutMS: 45000,
        });
        console.log('Connected to MongoDB');
    } catch (err) {
        console.error('MongoDB connection error:', err);
        // Don't exit the process, just log the error
        console.error('Continuing without database connection...');
    }
};

connectDB();

// Report Schema
const reportSchema = new mongoose.Schema({
    originalText: String,
    summary: String,
    analysis: {
        numericalData: {
            metrics: [{
                name: String,
                value: String,
                unit: String,
                normalRange: String,
                status: {
                    type: String,
                    enum: ['normal', 'warning', 'critical']
                },
                trend: {
                    type: String,
                    enum: ['increasing', 'decreasing', 'stable']
                }
            }],
            suggestedVisualizations: [{
                type: {
                    type: String,
                    enum: ['graph', 'chart', 'gauge']
                },
                title: String,
                description: String
            }]
        },
        keyFindings: [{
            finding: String,
            severity: {
                type: String,
                enum: ['normal', 'warning', 'critical']
            },
            category: String,
            explanation: String
        }],
        recommendations: [{
            recommendation: String,
            priority: {
                type: String,
                enum: ['high', 'medium', 'low']
            },
            timeframe: {
                type: String,
                enum: ['immediate', 'short-term', 'long-term']
            },
            rationale: String
        }],
        urgentConcerns: [{
            concern: String,
            action: String,
            impact: String
        }],
        simplifiedSummary: {
            mainPoints: [String],
            nextSteps: [String],
            medicalTerms: [{
                term: String,
                definition: String
            }]
        }
    },
    createdAt: { type: Date, default: Date.now }
});

const Report = mongoose.model('Report', reportSchema);

// Process report endpoint
app.post('/api/analyze-report', upload.single('report'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'No file uploaded' });
        }

        // Initialize Tesseract worker
        const worker = await createWorker();

        // Extract text from image
        const { data: { text } } = await worker.recognize(req.file.buffer);
        await worker.terminate();

        // Generate analysis using Groq
        const completion = await groq.chat.completions.create({
            messages: [
                {
                    role: "system",
                    content: `You are a medical expert AI assistant specialized in analyzing medical reports and presenting them in a visually appealing and easily understandable format. Your analysis should be detailed yet accessible to general users.

Key guidelines for your response:
1. Extract and highlight any numerical data that could be visualized (lab results, measurements, etc.)
2. Categorize findings by severity using a color-coding system:
   - Critical/Urgent (Red): Immediate attention needed
   - Warning (Yellow): Requires monitoring
   - Normal/Good (Green): Within healthy ranges
   - Informational (Blue): General information
3. Use clear hierarchical organization
4. Include trend analysis if temporal data is present
5. Provide context for medical terms`
                },
                {
                    role: "user",
                    content: `Please analyze this medical report and provide a structured response in the following JSON-like format:

{
    "numericalData": {
        "metrics": [
            {
                "name": "Metric Name",
                "value": "Numerical Value",
                "unit": "Unit of Measurement",
                "normalRange": "Reference Range",
                "status": "normal|warning|critical",
                "trend": "increasing|decreasing|stable" (if applicable)
            }
        ],
        "suggestedVisualizations": [
            {
                "type": "graph|chart|gauge",
                "title": "Visualization Title",
                "description": "What this visualization shows"
            }
        ]
    },
    "keyFindings": [
        {
            "finding": "Finding description",
            "severity": "normal|warning|critical",
            "category": "Category name",
            "explanation": "Simple explanation"
        }
    ],
    "recommendations": [
        {
            "recommendation": "Action item",
            "priority": "high|medium|low",
            "timeframe": "immediate|short-term|long-term",
            "rationale": "Why this is important"
        }
    ],
    "urgentConcerns": [
        {
            "concern": "Description of urgent issue",
            "action": "Required immediate action",
            "impact": "Potential consequences"
        }
    ],
    "simplifiedSummary": {
        "mainPoints": ["Point 1", "Point 2"],
        "nextSteps": ["Step 1", "Step 2"],
        "medicalTerms": [
            {
                "term": "Medical term",
                "definition": "Simple explanation"
            }
        ]
    }
}

Here's the report text: ${text}`
                }
            ],
            model: "mixtral-8x7b-32768",
            temperature: 0.5,
            max_tokens: 2048,
        });

        const analysis = completion.choices[0]?.message?.content;

        // Parse the JSON response
        const parsedAnalysis = JSON.parse(analysis);

        // Save to database with enhanced schema
        try {
            const report = new Report({
                originalText: text,
                summary: analysis,
                analysis: parsedAnalysis
            });
            await report.save();
        } catch (dbError) {
            console.error('Database save error:', dbError);
            // Continue without saving to database
        }

        res.json({
            success: true,
            text,
            analysis: parsedAnalysis
        });

    } catch (error) {
        console.error('Error processing report:', error);
        res.status(500).json({
            error: 'Error processing report',
            details: error.message
        });
    }
});

// Get report history
app.get('/api/reports', async (req, res) => {
    try {
        const reports = await Report.find().sort({ createdAt: -1 }).limit(10);
        res.json(reports);
    } catch (error) {
        res.status(500).json({ error: 'Error fetching reports' });
    }
});

// Chatbot endpoint for health-related queries
app.get('/api/chat', (req, res) => {
    // Simple health check endpoint
    res.status(200).json({ status: 'ok' });
});

app.post('/api/chat', async (req, res) => {
    try {
        const { message } = req.body;
        
        if (!message) {
            return res.status(400).json({ 
                success: false,
                error: 'Message is required' 
            });
        }

        console.log(`Received message for chatbot: ${message}`);

        const completion = await groq.chat.completions.create({
            messages: [
                {
                    role: "system",
                    content: "You are a medical expert AI assistant specialized in explaining frequently used medical terms in layman's terms to the user. You should provide clear, accurate, and easy-to-understand explanations while maintaining medical accuracy. If the query is not health-related, politely inform the user that you can only assist with health-related questions."
                },
                {
                    role: "user",
                    content: message
                }
            ],
            model: "mixtral-8x7b-32768",
            temperature: 0.7,
            max_tokens: 1024,
        });

        const response = completion.choices[0]?.message?.content || "Sorry, I couldn't process your request.";

        res.json({
            success: true,
            response
        });

    } catch (error) {
        console.error('Error in chatbot:', error);
        res.status(500).json({
            success: false,
            error: 'Error processing chat message',
            response: "Sorry, there was an error with the chatbot. Please try again later."
        });
    }
});

// Error handling middleware
app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(500).json({
        error: 'Something went wrong!',
        details: err.message
    });
});

app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});
