# 🎯 AI Men's Hairstyle Recommender & Visualizer

A sophisticated full-stack web application that uses AI to analyze facial features and recommend the perfect men's hairstyles with virtual try-on capabilities. **Now powered by Google Gemini LLM for intelligent, personalized recommendations!**

## ✨ Features

### 🤖 **NEW: AI-Powered Intelligent Recommendations**
- **Google Gemini LLM Integration** - Advanced AI-powered hairstyle recommendations
- **Personalized Analysis** - Considers age, race, gender, face shape, and hair texture
- **Cultural Awareness** - Culturally appropriate and authentic styling suggestions
- **Dynamic Generation** - Creates custom hairstyle recommendations in real-time
- **Smart Fallback** - Automatically falls back to rule-based system if LLM unavailable

### 🔍 **Intelligent Face Analysis**
- Advanced facial landmark detection using OpenCV
- CNN-based face shape classification optimized for men
- Real-time facial feature analysis and visualization
- High-accuracy face shape determination (Oval, Round, Square, Heart, Long, Diamond)

### 💼 **Men's Hairstyle Recommendations**
- **LLM-Generated Recommendations** - Personalized suggestions based on multiple factors
- Curated database of popular men's hairstyles (fade, undercut, pompadour, crew cut, etc.)
- Face shape-based intelligent recommendations
- Professional and modern styling options
- Detailed descriptions and suitability information

### 🎨 **AI-Powered Virtual Try-On**
- Generate realistic previews using Stability AI or OpenAI DALL-E
- See how different men's hairstyles look on your face
- High-quality image generation with styling prompts
- Download and share your new look

### 🎯 **Modern Masculine Design**
- Professional navy blue and steel gray color scheme
- Clean, modern interface optimized for men
- Mobile-responsive design with intuitive workflow
- Progress tracking through 4-step process

## 🛠 Tech Stack

### Backend (Python)
- **FastAPI** - High-performance web framework
- **Google Gemini LLM** - Advanced AI recommendations via LangChain
- **LangChain** - LLM integration framework
- **OpenCV** - Computer vision and facial analysis
- **PyTorch** - Deep learning for face classification
- **Pydantic** - Data validation and serialization
- **PIL/Pillow** - Image processing

### Frontend (Next.js)
- **Next.js 14** - React framework with App Router
- **TypeScript** - Type-safe development
- **Tailwind CSS** - Utility-first styling with masculine color palette
- **React Hooks** - Modern state management

### AI Integration
- **Google Gemini Pro** - Intelligent hairstyle recommendations
- **Stability AI** - Professional image generation
- **OpenAI DALL-E** - Alternative AI image generation
- **Custom CNN** - Face shape classification model

## 🚀 Getting Started

### Prerequisites
- **Node.js** 18+ and npm
- **Python** 3.8+
- **Git** for version control
- **Google Gemini API Key** (for AI recommendations)

### 1. Clone the Repository
```bash
git clone <repository-url>
cd AI-Mens-Hairstyle-Recommender
```

### 2. Set Up Backend
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install Python dependencies
pip install -r backend/requirements.txt
```

### 3. Configure API Keys
Create a `.env` file in the `backend/` directory:

```bash
# Required for AI-powered recommendations
GEMINI_API_KEY=your_google_gemini_api_key_here

# Optional for image generation
STABILITY_AI_API_KEY=your_stability_ai_key_here
OPENAI_API_KEY=your_openai_key_here

# Development settings
DEBUG=True
```

**Getting Google Gemini API Key:**
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Create a new API key
4. Copy the key to your `.env` file

### 4. Set Up Frontend
```bash
cd frontend
npm install

# Create .env.local file:
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### 5. Run the Application

**Terminal 1 - Backend:**
```bash
# From backend directory
# On Windows:
set GEMINI_API_KEY=your_key_here && python app/main.py
# On macOS/Linux:
export GEMINI_API_KEY=your_key_here && python app/main.py
# Backend runs on http://localhost:8000
```

**Terminal 2 - Frontend:**
```bash
# From frontend directory
npm run dev
# Frontend runs on http://localhost:3000
```

### 6. Access the App
Open your browser and navigate to `http://localhost:3000`

## 🧪 Testing the LLM Integration

To verify the AI integration is working:

```bash
# From backend directory
python test_api.py
```

This will test:
- Server health and LLM status
- AI-powered recommendation generation
- API endpoint functionality

## 📱 How It Works

### Step 1: Create Profile (Enhanced)
- Enter demographic information (age, ethnicity, gender)
- Specify hair texture preferences
- AI uses this data for personalized recommendations

### Step 2: Upload Photo
- Drag & drop or select a clear front-facing photo
- Supports JPG, PNG, WebP formats
- Automatic image validation and processing

### Step 3: Face Analysis
- AI detects facial landmarks and features
- Determines face shape with confidence scoring
- Visualizes detected points on your photo

### Step 4: AI Recommendations (NEW)
- **Google Gemini analyzes your complete profile**
- Considers face shape, age, ethnicity, and preferences
- Generates culturally-aware, personalized suggestions
- Shows 6 diverse, AI-curated hairstyles

### Step 5: Virtual Try-On
- AI generates realistic preview of chosen hairstyle
- Download or share your new look
- Try multiple styles with easy regeneration

## 🎨 AI-Generated Hairstyle Recommendations

The Gemini LLM considers:

- **Face Shape** - Oval, round, square, heart, long proportions
- **Age** - Age-appropriate styling suggestions
- **Ethnicity** - Culturally authentic and appropriate styles
- **Hair Texture** - Straight, wavy, curly, coily compatibility
- **Lifestyle** - Professional, casual, trendy preferences
- **Maintenance** - Low to high-maintenance options

Sample AI-generated styles include:
- **Professional Fade Variations** - Tailored to face shape
- **Cultural Authenticity** - Styles respecting heritage
- **Age-Appropriate Trends** - Modern but suitable
- **Texture-Specific Cuts** - Optimized for hair type
- **Lifestyle Matching** - Professional vs. casual options

## 🔧 Configuration

### LLM Prompt Customization
Modify the prompt template in `backend/services/llm_hairstyle_recommender.py`:

```python
def _setup_prompt_template(self):
    # Customize the prompt for different recommendation styles
    template = """You are a professional hairstylist AI assistant..."""
```

### Fallback Behavior
The system automatically:
- Uses traditional recommendations if Gemini API is unavailable
- Provides graceful degradation of service
- Logs LLM status in health check endpoint

### Traditional Mode
To disable LLM and use rule-based recommendations:
- Remove or comment out `GEMINI_API_KEY` in `.env`
- The system will automatically use the traditional recommender

## 📝 API Endpoints

### Enhanced Endpoints
- `GET /` - Health check with LLM status
- `POST /recommend-hairstyles/` - AI-powered or traditional recommendations
- `POST /analyze-face/` - Facial analysis with profile integration
- `POST /generate-hairstyle-image/` - AI image generation
- `POST /create-profile/` - User profile management

### New Response Format
```json
{
  "status": "healthy",
  "service": "AI Hairstyle Recommender & Visualizer", 
  "version": "2.1.0",
  "llm_enabled": true,
  "features": [
    "Google Gemini AI-powered recommendations",
    "Personalized styling based on age, race, and features",
    "Cultural-aware hairstyle suggestions",
    "Dynamic hairstyle generation"
  ]
}
```

## 🎯 System Architecture

```
Frontend (Next.js) 
    ↓ API Calls
Backend (FastAPI)
    ↓ Face Analysis
Face Classifier (VGG16)
    ↓ User Profile
LLM Service (Google Gemini)
    ↓ Recommendations
Image Generator (Stability AI/OpenAI)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Test with both LLM and traditional modes
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## 🔒 Privacy & Security

- API keys stored securely in environment variables
- User profiles stored temporarily (consider database for production)
- Face images processed locally, not stored permanently
- Google Gemini API calls follow Google's privacy policies

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Google for the Gemini LLM API and AI capabilities
- LangChain for seamless LLM integration
- OpenCV community for computer vision tools
- Stability AI and OpenAI for image generation capabilities
- Men's grooming and styling communities for hairstyle insights

---

**Ready to experience AI-powered hairstyle recommendations? Start your transformation at `http://localhost:3000`** 🤖✂️

### 🚀 New in v2.1.0
- Google Gemini LLM integration for intelligent recommendations
- Personalized analysis based on age, race, and preferences
- Cultural awareness in hairstyle suggestions
- Dynamic AI-generated hairstyle options
- Enhanced user profiling system
- Graceful fallback to traditional recommendations 