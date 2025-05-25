# 🎯 AI Men's Hairstyle Recommender & Visualizer

A sophisticated full-stack web application that uses AI to analyze facial features and recommend the perfect men's hairstyles with virtual try-on capabilities.

## ✨ Features

### 🔍 **Intelligent Face Analysis**
- Advanced facial landmark detection using OpenCV
- CNN-based face shape classification optimized for men
- Real-time facial feature analysis and visualization
- High-accuracy face shape determination (Oval, Round, Square, Heart, Long, Diamond)

### 💼 **Men's Hairstyle Recommendations**
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
- **Stability AI** - Professional image generation
- **OpenAI DALL-E** - Alternative AI image generation
- **Custom CNN** - Face shape classification model

## 🚀 Getting Started

### Prerequisites
- **Node.js** 18+ and npm
- **Python** 3.8+
- **Git** for version control

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
pip install -r requirements.txt

# Set environment variables (optional)
# Create .env file in backend/ directory:
STABILITY_AI_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
```

### 3. Set Up Frontend
```bash
cd frontend
npm install

# Create .env.local file:
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### 4. Run the Application

**Terminal 1 - Backend:**
```bash
# From project root
export PYTHONPATH=. && python app/main.py
# Backend runs on http://localhost:8000
```

**Terminal 2 - Frontend:**
```bash
# From frontend directory
npm run dev
# Frontend runs on http://localhost:3000
```

### 5. Access the App
Open your browser and navigate to `http://localhost:3000`

## 📱 How It Works

### Step 1: Upload Photo
- Drag & drop or select a clear front-facing photo
- Supports JPG, PNG, WebP formats
- Automatic image validation and processing

### Step 2: Face Analysis
- AI detects facial landmarks and features
- Determines face shape with confidence scoring
- Visualizes detected points on your photo

### Step 3: Style Recommendations
- Shows men's hairstyles suited to your face shape
- Browse options like fades, undercuts, pompadours
- Select your preferred style for virtual try-on

### Step 4: Virtual Try-On
- AI generates realistic preview of chosen hairstyle
- Download or share your new look
- Try multiple styles with easy regeneration

## 🎨 Men's Hairstyle Collection

Our curated selection includes:

- **Classic Fade** - Timeless professional cut
- **Modern Undercut** - Bold and edgy styling
- **Crew Cut** - Military-inspired practicality
- **Pompadour** - Sophisticated retro style
- **Buzz Cut** - Ultra-low maintenance
- **Quiff** - Textured modern volume
- **Side Part** - Traditional gentleman's cut
- **Textured Crop** - Casual contemporary style
- **Slicked Back** - Formal business look
- **Caesar Cut** - Forward-styled classic

## 🔧 Configuration

### AI Service Setup (Optional)
The app works without API keys using fallback modes:

**Stability AI:**
```bash
export STABILITY_AI_API_KEY=your_key_here
```

**OpenAI:**
```bash
export OPENAI_API_KEY=your_key_here
```

### Customization
- Modify `backend/data/hairstyles.json` to add new men's styles
- Update color scheme in `frontend/tailwind.config.ts`
- Adjust AI prompts in service files

## 📝 API Endpoints

- `POST /analyze-face/` - Facial analysis and landmarks
- `POST /recommend-hairstyles/` - Get style recommendations
- `POST /generate-hairstyle-image/` - AI image generation
- `GET /` - Health check

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenCV community for computer vision tools
- Stability AI and OpenAI for image generation capabilities
- Men's grooming and styling communities for hairstyle insights
- Modern web development ecosystem (Next.js, FastAPI, Tailwind)

---

**Ready to find your perfect men's hairstyle? Start your transformation at `http://localhost:3000`** 🎯✂️ 