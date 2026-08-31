# Introducing ProStock AI: Indian Stock Price Prediction with LSTM Deep Learning

## I'm Building a Professional AI Model for Indian Stock Market Predictions

Hello everyone! I'm excited to share a project I've been working on - ProStock AI, a professional LSTM deep learning model for predicting NSE Indian stock movements.

## The Problem I'm Solving

Indian stock traders face unique challenges:
- **Diverse price scales**: Stocks range from ₹100 to ₹10,000+ per share
- **Market volatility**: Indian markets have unique patterns and trends
- **Limited tools**: Few AI/ML tools are optimized for Indian (NSE) stocks
- **Global macro factors**: Indian stocks are affected by Nifty 50, USD/INR, S&P 500, and more

## The Solution: ProStock AI

**ProStock AI** is an ensemble LSTM model that combines:
- **Single-stock LSTM**: Trained on individual stock historical data
- **Universal market bias**: Accounts for Nifty 50, S&P 500, and USD/INR trends
- **News sentiment analysis**: Uses TextBlob to analyze market-moving news
- **11 technical features**: Including moving averages, RSI, volatility, and relative volume

## Key Features

### 🤖 Multi-Path Ensemble Prediction
- Single-stock model + universal bias combination
- News sentiment integration (Positive 🟢 / Neutral ⚪ / Negative 🔴)
- Global macro indicators for context

### 📈 Multi-Horizon Forecasts
- **1-day**: Short-term trading decisions
- **5-day**: Week-long trend predictions
- **10-day**: Mid-term strategy planning
- **20-day**: Long-term investment outlook

### 🛡️ Risk Management
- **95% VaR (Value at Risk)**: Probability of loss
- **Stop Loss**: Automatic exit levels
- **Take Profit**: Target price levels
- **Annual Volatility**: Risk metric for position sizing

### 📊 Feature Set (11 Input Features)
1. Close price
2. Nifty 50 index
3. USD/INR exchange rate
4. S&P 500 index
5. 5-day Moving Average (MA5)
6. 20-day Moving Average (MA20)
7. Return percentage
8. Volatility
9. Relative Strength Index (RSI)
10. Relative Volume
11. Price-to-Volume ratio

### 🏢 Company Information
- Company name and sector
- Industry classification
- Market capitalization

## Technology Stack

- **Deep Learning**: LSTM (Long Short-Term Memory) networks
- **Data**: Yahoo Finance (yfinance) for historical prices
- **Features**: Technical indicators, macro data, news sentiment
- **Framework**: TensorFlow/Keras
- **Web API**: FastAPI
- **UI**: Tailwind CSS with dark-mode glassmorphism design

## Current Progress

✅ **Model trained**: Loss: 4.65e-04 on training data
✅ **Walk-forward validation**: 3/3 folds, 66.67% directional accuracy
✅ **Web UI**: Running at localhost:8000 with full feature set
✅ **All features implemented**: Multi-horizon, risk metrics, sentiment
✅ **Git repository**: Initial commit with 16 essential files
✅ **Hugging Face ready**: Model upload script prepared
✅ **Sponsorship**: funding.yml configured with multiple platforms

## Why Support This Project?

By supporting ProStock AI on Patreon, you're helping:

1. **Continue development**: More time for model improvements
2. **Expand coverage**: Add more NSE tickers (currently 15,771 available)
3. **Enhance features**: Additional technical indicators and better sentiment analysis
4. **Free tier access**: Patreon supporters get early access to new features
5. **Transparent progress**: Regular updates on model performance and accuracy

## Patreon Tiers (Planned)

- **Supporter ($5/month)**: Access to prediction API, monthly model updates
- **Analyst ($15/month)**: All supporter benefits + personalized stock analysis
- **Quant ($30/month)**: All analyst benefits + priority feature requests, beta access
- **Sponsor ($50/month)**: All quant benefits + consulting session, model customization

## How You Can Help

### 💡 Spread the Word
- Share with fellow traders and investors
- Star the GitHub repository
- Follow on social media

### 🤝 Contribute
- Sponsor on Patreon or GitHub Sponsors
- Provide feedback on features
- Report bugs or suggest improvements

### 📬 Stay Updated
- Follow the GitHub repository for updates
- Subscribe to project announcements

## Recent Milestones

- **Day 1**: Initial LSTM model architecture
- **Day 7**: Ensemble approach with universal bias
- **Day 14**: Multi-horizon predictions implemented
- **Day 21**: News sentiment analysis added
- **Day 30**: Walk-forward cross-validation (66.67% accuracy)
- **Day 45**: Web UI with Tailwind CSS design
- **Day 60**: Hugging Face model card prepared
- **Day 75**: Funding configuration (funding.yml)

## Get Involved

🔗 **GitHub**: https://github.com/samarulraj/indian-stock-lstm
🔗 **Repository**: samarulraj/indian-stock-lstm
🔗 **Model**: Available on Hugging Face

---

*ProStock AI is an ongoing project. Your support helps make Indian stock market AI prediction accessible to everyone.*

*Built with ❤️ for the Indian trading community*
