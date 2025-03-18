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
        // Validate file upload
        if (!req.file) {
            return res.status(400).json({ 
                success: false,
                error: 'No file uploaded' 
            });
        }

        console.log('File received:', {
            filename: req.file.originalname,
            size: req.file.size,
            mimetype: req.file.mimetype
        });

        // Initialize Tesseract worker with error handling
        let worker;
        try {
            worker = await createWorker();
            console.log('Tesseract worker initialized');
        } catch (tesseractError) {
            console.error('Tesseract initialization error:', tesseractError);
            return res.status(500).json({
                success: false,
                error: 'Failed to initialize text recognition',
                details: tesseractError.message
            });
        }

        // Extract text from image with error handling
        let text;
        try {
            const result = await worker.recognize(req.file.buffer);
            text = result.data.text;
            console.log('Text extracted successfully, length:', text.length);
            await worker.terminate();
        } catch (ocrError) {
            console.error('OCR processing error:', ocrError);
            await worker.terminate();
            return res.status(500).json({
                success: false,
                error: 'Failed to process image',
                details: ocrError.message
            });
        }

        if (!text || text.trim().length === 0) {
            return res.status(400).json({
                success: false,
                error: 'No text could be extracted from the image'
            });
        }

        // Generate analysis using Groq with error handling
        let analysis;
        try {
            const completion = await groq.chat.completions.create({
                messages: [
                    {
                        role: "system",
                        content: `You are a medical expert AI assistant that MUST respond with valid JSON. Your task is to analyze medical reports and present them in a structured format.

Key requirements:
1. ALWAYS respond with properly formatted JSON
2. Use the exact structure provided in the example
3. Do not include any text outside the JSON structure
4. Ensure all JSON values are properly quoted
5. Keep responses concise to avoid token limits`
                    },
                    {
                        role: "user",
                        content: `Analyze this medical report and respond ONLY with a JSON object in this exact structure:

{
    "numericalData": {
        "metrics": [],
        "suggestedVisualizations": []
    },
    "keyFindings": [],
    "recommendations": [],
    "urgentConcerns": [],
    "simplifiedSummary": {
        "mainPoints": [],
        "nextSteps": [],
        "medicalTerms": []
    }
}

If no data is found for a section, keep it as an empty array. Here's the report text: ${text}`
                    }
                ],
                model: "mixtral-8x7b-32768",
                temperature: 0.3, // Reduced temperature for more consistent output
                max_tokens: 2048,
            });

            if (!completion.choices?.[0]?.message?.content) {
                throw new Error('Empty response from Groq API');
            }

            analysis = completion.choices[0].message.content.trim();
            
            // Validate that the response starts and ends with curly braces
            if (!analysis.startsWith('{') || !analysis.endsWith('}')) {
                throw new Error('Invalid JSON structure in API response');
            }

            console.log('Analysis generated successfully');
            console.log('Raw analysis:', analysis); // Log the raw response for debugging
        } catch (groqError) {
            console.error('Groq API error:', groqError);
            return res.status(500).json({
                success: false,
                error: 'Failed to analyze report',
                details: groqError.message
            });
        }

        // Parse the JSON response with error handling and validation
        let parsedAnalysis;
        try {
            // First try to parse the JSON
            parsedAnalysis = JSON.parse(analysis);

            // Validate the required structure
            const requiredFields = ['numericalData', 'keyFindings', 'recommendations', 'urgentConcerns', 'simplifiedSummary'];
            const missingFields = requiredFields.filter(field => !(field in parsedAnalysis));

            if (missingFields.length > 0) {
                throw new Error(`Missing required fields: ${missingFields.join(', ')}`);
            }

            // Ensure arrays are present even if empty
            parsedAnalysis.numericalData = parsedAnalysis.numericalData || { metrics: [], suggestedVisualizations: [] };
            parsedAnalysis.numericalData.metrics = parsedAnalysis.numericalData.metrics || [];
            parsedAnalysis.numericalData.suggestedVisualizations = parsedAnalysis.numericalData.suggestedVisualizations || [];
            parsedAnalysis.keyFindings = parsedAnalysis.keyFindings || [];
            parsedAnalysis.recommendations = parsedAnalysis.recommendations || [];
            parsedAnalysis.urgentConcerns = parsedAnalysis.urgentConcerns || [];
            parsedAnalysis.simplifiedSummary = parsedAnalysis.simplifiedSummary || {
                mainPoints: [],
                nextSteps: [],
                medicalTerms: []
            };

            console.log('Analysis parsed and validated successfully');
        } catch (parseError) {
            console.error('JSON parsing or validation error:', parseError);
            console.error('Raw analysis:', analysis);
            
            // Attempt to create a basic valid response
            parsedAnalysis = {
                numericalData: {
                    metrics: [],
                    suggestedVisualizations: []
                },
                keyFindings: [{
                    finding: "Error processing report",
                    severity: "warning",
                    category: "System Error",
                    explanation: "The system encountered an error while analyzing the report. Please try again or contact support."
                }],
                recommendations: [],
                urgentConcerns: [],
                simplifiedSummary: {
                    mainPoints: ["Error in report analysis"],
                    nextSteps: ["Please try uploading the report again"],
                    medicalTerms: []
                }
            };

            // Still return success but with the error structure
            return res.status(200).json({
                success: true,
                text,
                analysis: parsedAnalysis,
                warning: 'Error parsing AI response, showing fallback analysis'
            });
        }

        // Save to database with enhanced schema
        try {
            const report = new Report({
                originalText: text,
                summary: analysis,
                analysis: parsedAnalysis
            });
            await report.save();
            console.log('Report saved to database');
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
        console.error('Unhandled error in analyze-report:', error);
        res.status(500).json({
            success: false,
            error: 'Error processing report',
            details: error.message,
            stack: process.env.NODE_ENV === 'development' ? error.stack : undefined
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
