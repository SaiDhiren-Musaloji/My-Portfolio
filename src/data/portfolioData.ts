import { Project, Certificate, Skill, ContactInfo } from '../types';

export const projects: Project[] = [
  {
    id: 1,
    title: "AI Research Assistant with Retrieval-Augmented Generation",
    description: "• Traditional research assistants lack contextual understanding and fail to provide accurate, up-to-date information from multiple sources with proper citation and verification\n\n• Developed intelligent research assistant integrating Google Gemini 1.5 Flash API with RAG, building custom vector database using scikit-learn's NearestNeighbors, implementing newspaper3k for article extraction, and engineering end-to-end workflow with multi-model NLP processing (BART, BERT, DistilBERT) for contextually-grounded responses.",
    image: "/images/projects/Rag.png",
    technologies: ["Python", "Google Gemini API", "BART", "BERT", "DistilBERT", "RAG", "NLP", "newspaper3k", "scikit-learn", "Vector Database"],
    githubUrl: "https://github.com/yourusername/ai-research-assistant",
    category: "Generative AI",
    detailedDescription: "Developed an intelligent research assistant integrating Google Gemini 1.5 Flash API with retrieval-augmented generation, combining web search capabilities with contextual document processing. Built custom data processing pipeline integrating newspaper3k for article extraction, serper.dev for search results, with custom error handling, rate limiting, and multi-model NLP processing (BART, BERT, DistilBERT). Implemented custom vector database solution from scratch using scikit-learn's NearestNeighbors and numpy arrays for in-memory semantic similarity search and document retrieval. Engineered end-to-end RAG workflow orchestrating pre-trained transformer models for embedding generation, custom retrieval logic, and dynamic prompt construction for contextually-grounded responses. Designed modular architecture with asynchronous processing, session management, and multi-model integration for summarization, named entity recognition, and sentiment analysis."
  },
  {
    id: 2,
    title: "Software License Compliance Analysis and Cost Optimization System",
    description: "Comprehensive compliance analytics solution with multi-source data extraction and cost optimization for enterprise software management.",
    image: "/images/projects/SC.webp",
    technologies: ["SQL", "Power BI", "Excel", "Python", "Data Analysis", "Cost Optimization"],
    githubUrl: "https://github.com/yourusername/license-compliance-system",
    category: "Data Analysis",
    detailedDescription: "Built a comprehensive software license compliance and cost optimization system for enterprise environments. The solution analyzes software usage patterns, identifies compliance risks, and provides actionable insights for cost reduction."
  },
  {
    id: 3,
    title: "Adversarial Attack on CIFAR-10 Models",
    description: "• Minor, imperceptible perturbations to input images can fool state-of-the-art deep learning models into making incorrect predictions, highlighting critical security vulnerabilities in AI systems\n\n• Implemented ensemble-based adversarial attacks using Iterative Fast Gradient Sign Method (I-FGSM) with gradient aggregation across multiple pre-trained models, achieving 85% attack success rate while maintaining visual imperceptibility.",
    image: "/images/projects/Adversarial.png",
    technologies: ["Python", "PyTorch", "TensorFlow", "CIFAR-10", "Adversarial ML", "Computer Vision"],
    githubUrl: "https://github.com/yourusername/adversarial-attacks",
    category: "ML Modeling",
    detailedDescription: "Implemented ensemble-based adversarial attacks using Iterative Fast Gradient Sign Method (I-FGSM) with gradient aggregation across multiple pre-trained models, achieving 85% attack success rate while maintaining visual imperceptibility."
  },
  {
    id: 4,
    title: "BERT-Based Reading Comprehension",
    description: "• Traditional question answering systems struggle to understand contextual relationships and accurately extract answer spans from given passages\n\n• Fine-tuned BERT model with custom loss functions for start/end position prediction, implementing dynamic learning rate scheduling and gradient accumulation techniques for improved span extraction accuracy.",
    image: "/images/projects/bert.png",
    technologies: ["Python", "BERT", "Transformers", "PyTorch", "NLP", "Question Answering"],
    githubUrl: "https://github.com/yourusername/bert-reading-comprehension",
    category: "Generative AI"
  },
  {
    id: 5,
    title: "Deep Reinforcement Learning for Lunar Lander",
    description: "• Learning complex sequential decision-making under uncertainty where an agent must balance multiple competing objectives including safe landing, fuel efficiency, and trajectory optimization in continuous action spaces\n\n• Implemented Actor-Critic architecture with policy gradient methods, custom reward shaping, and experience replay mechanisms achieving consistent successful landings with scores above 200 points.",
    image: "/images/projects/lunarlander.png",
    technologies: ["Python", "PyTorch", "OpenAI Gym", "Reinforcement Learning", "Actor-Critic"],
    githubUrl: "https://github.com/yourusername/lunar-lander-rl",
    category: "ML Modeling"
  },
  {
    id: 6,
    title: "Graph Neural Networks for Financial Prediction",
    description: "• Traditional time series models fail to capture complex inter-asset relationships and network effects that significantly influence financial market dynamics\n\n• Implemented Graph Attention Networks (GAT) to model both spatial relationships between assets and temporal dependencies, achieving superior prediction accuracy through ensemble strategies and multi-head attention mechanisms.",
    image: "/images/projects/deep_learning.png",
    technologies: ["Python", "PyTorch", "Graph Neural Networks", "GAT", "Financial ML"],
    githubUrl: "https://github.com/yourusername/gnn-financial-prediction",
    category: "ML Modeling"
  },
  {
    id: 7,
    title: "Transformer-Based Speaker Classification",
    description: "• Traditional RNN-based approaches struggle to capture long-range dependencies in audio sequences and fail to effectively classify speakers from variable-length MFCC features across 600 different speakers\n\n• Implemented transformer encoder architecture with custom positional encoding for audio data, achieving superior accuracy through multi-head self-attention mechanisms and mixed precision training.",
    image: "/images/projects/Transformer-Based Speaker Classification.png",
    technologies: ["Python", "PyTorch", "Transformers", "Audio Processing", "MFCC"],
    githubUrl: "https://github.com/yourusername/speaker-classification",
    category: "ML Modeling"
  },
  {
    id: 8,
    title: "Self-Supervised Learning for User Localization",
    description: "• Traditional wireless localization methods require extensive labeled data and fail to extract meaningful representations from complex channel state information for accurate 3D position prediction\n\n• Implemented two-stage self-supervised approach using autoencoder for feature extraction followed by position prediction model, leveraging unlabeled channel data to achieve accurate localization with limited labeled samples.",
    image: "/images/projects/Self-Supervised Learning for User Localization.jpg",
    technologies: ["Python", "PyTorch", "Self-Supervised Learning", "Autoencoders", "Wireless ML"],
    githubUrl: "https://github.com/yourusername/user-localization",
    category: "ML Modeling"
  },
  {
    id: 9,
    title: "Neural Machine Translation with Transformers",
    description: "• Traditional RNN-based translation systems struggle with long-range dependencies, parallel processing limitations, and maintaining contextual accuracy across variable-length sequences in English to Traditional Chinese translation\n\n• Implemented transformer encoder-decoder architecture with SentencePiece tokenization, custom beam search decoding, and label smoothing regularization achieving superior translation quality and reduced training time.",
    image: "/images/projects/bert.png",
    technologies: ["Python", "PyTorch", "Transformers", "SentencePiece", "NMT"],
    githubUrl: "https://github.com/yourusername/neural-translation",
    category: "Generative AI"
  },
  {
    id: 10,
    title: "Deep Neural Networks for Speech Classification",
    description: "• Traditional machine learning approaches fail to capture complex non-linear relationships in acoustic feature space and struggle with inherent variability in speech patterns across different speakers and contexts for phoneme classification\n\n• Implemented multi-layer neural network with context window concatenation, batch normalization, and advanced data augmentation techniques achieving substantial accuracy improvements over baseline methods.",
    image: "/images/projects/Deep Neural Networks for Speech Classification.webp",
    technologies: ["Python", "PyTorch", "Speech Processing", "MFCC", "Phoneme Classification"],
    githubUrl: "https://github.com/yourusername/speech-classification",
    category: "ML Modeling"
  },
  {
    id: 11,
    title: "Handwritten Telugu Character Recognition",
    description: "• Traditional OCR systems struggle with complex Indic scripts like Telugu due to character variations, ligatures, and contextual dependencies\n\n• Implemented CNN-based OCR framework with comprehensive data preprocessing, custom character segmentation, and multi-scale feature extraction achieving robust recognition accuracy for Telugu script.",
    image: "/images/projects/telugu handwritten recognition.png",
    technologies: ["TensorFlow", "Keras", "CNN", "OpenCV", "OCR"],
    githubUrl: "https://github.com/yourusername/telugu-ocr",
    category: "ML Modeling"
  },
  {
    id: 12,
    title: "Distributed Recognition Engine Using Cloud-Native Paradigms",
    description: "Problem: Traditional image and text recognition systems lack scalability and reliability for high-volume processing with varying workloads. Solution: Built asynchronous recognition pipeline using AWS EC2, SQS, S3, and Rekognition with stateless compute strategies, achieving 99.9% uptime through visibility timeout calibration and retry mechanisms.",
    image: "/images/projects/sqs.png",
    technologies: ["AWS", "Docker", "Kubernetes", "Terraform", "Cloud Computing"],
    githubUrl: "https://github.com/yourusername/aws-optimization",
    category: "Engineering Solutions"
  },
  {
    id: 13,
    title: "Spatiotemporal Forecasting of Urban Traffic Networks",
    description: "Problem: Traditional traffic prediction models fail to capture complex spatiotemporal dependencies and topological interactions between network nodes in urban traffic systems. Solution: Engineered fusion architecture combining ARIMA, LSTM, and GNN models with PCA and t-SNE dimensionality reduction, achieving robust multi-horizon prediction across 325 sensors in the PEMS-BAY dataset.",
    image: "/images/projects/traffic_forecasting.png",
    technologies: ["Python", "ARIMA", "LSTM", "GNN", "PCA", "t-SNE", "Time Series"],
    githubUrl: "https://github.com/yourusername/time-series-analysis",
    category: "Data Analysis"
  },
  {
    id: 14,
    title: "High-Availability Financial Analytics System",
    description: "Problem: Traditional banking analytics systems lack real-time processing capabilities and fail to handle high concurrency requirements for fraud detection and transactional analysis. Solution: Developed transactional analytics platform with responsive dashboard, anomaly detection, and time-series mapping, containerized using Docker with CI/CD pipelines for high-availability deployment.",
    image: "/images/projects/database.png",
    technologies: ["Python", "Docker", "CI/CD", "Anomaly Detection", "Time Series", "Financial Analytics"],
    githubUrl: "https://github.com/yourusername/database-design",
    category: "Data Engineering"
  },
  {
    id: 15,
    title: "Parallelized ML Pipeline for Oenological Forecasting",
    description: "Problem: Traditional wine quality prediction models lack scalability and fail to handle large-scale datasets efficiently, resulting in slow training times and limited throughput for production inference. Solution: Created Spark-based distributed system on AWS EMR with partitioned data and in-memory caching, achieving 60% training speed improvement and high-throughput inference with autoscaling and performance logging.",
    image: "/images/projects/Parallel.png",
    technologies: ["Apache Spark", "AWS EMR", "ML Pipeline", "Distributed Computing", "Wine Quality Prediction"],
    githubUrl: "https://github.com/yourusername/parallel-computing",
    category: "Engineering Solutions"
  },
  {
    id: 16,
    title: "Flight Data Analysis with Scalability Testing",
    description: "Problem: Traditional flight data analysis systems struggle with processing large-scale historical datasets and fail to provide scalable insights across multi-year timeframes. Solution: Developed AWS-based MapReduce pipelines orchestrated with Apache Oozie for distributed flight data processing, quantifying performance metrics and scalability across multi-year datasets from 1987 to 2008.",
    image: "/images/projects/flight_data.png",
    technologies: ["AWS", "MapReduce", "Apache Oozie", "Big Data", "Data Analysis", "Scalability Testing"],
    githubUrl: "https://github.com/yourusername/flight-analytics",
    category: "Data Analysis"
  },
  {
    id: 17,
    title: "Equity Portfolio Management",
    description: "Problem: Traditional portfolio management approaches lack data-driven decision making and fail to optimize position sizing and risk management for algorithmic trading strategies. Solution: Formulated data-driven equity strategies using Python libraries (Pandas, NumPy, Matplotlib) to optimize a $5M portfolio, deploying quantitative analysis and risk modeling techniques for algorithmic position sizing and performance benchmarking.",
    image: "/images/projects/portfolio.png",
    technologies: ["Python", "Pandas", "NumPy", "Matplotlib", "Quantitative Analysis", "Risk Modeling", "Algorithmic Trading"],
    githubUrl: "https://github.com/yourusername/portfolio-website",
    category: "Engineering Solutions"
  },

];

export const certificates: Certificate[] = [
  {
    id: 1,
    title: "Azure AI Engineer",
    issuer: "Microsoft",
    image: "/images/certificates/azure_ai_engineer.png",
    pdfUrl: "/images/certificates/AI.pdf",
    issueDate: "2022"
  },
  {
    id: 2,
    title: "Azure Data Scientist",
    issuer: "Microsoft",
    image: "/images/certificates/azure_data_scientist.png",
    pdfUrl: "/images/certificates/DS.pdf",
    issueDate: "2022"
  },
  {
    id: 3,
    title: "GCP Vertex AI",
    issuer: "Google Cloud",
    image: "/images/certificates/gcp prompt.png",
    issueDate: "2024"
  },
  {
    id: 4,
    title: "Databricks Gen AI",
    issuer: "Databricks",
    image: "/images/certificates/databricks.png",
    issueDate: "2025"
  },
  {
    id: 5,
    title: "Databricks Fundamentals",
    issuer: "Databricks",
    image: "/images/certificates/databrick.png",
    issueDate: "2025"
  }
];

export const skills: Skill[] = [
  // Machine Learning
  { id: 1, name: "Scikit-learn", category: "machine-learning", proficiency: 88 },
  { id: 2, name: "Pandas", category: "machine-learning", proficiency: 95 },
  { id: 3, name: "NumPy", category: "machine-learning", proficiency: 92 },
  { id: 4, name: "Power BI", category: "machine-learning", proficiency: 80 },
  { id: 5, name: "R", category: "machine-learning", proficiency: 70 },
  
  // Deep Learning
  { id: 6, name: "TensorFlow", category: "deep-learning", proficiency: 90 },
  { id: 7, name: "PyTorch", category: "deep-learning", proficiency: 92 },
  { id: 8, name: "BERT", category: "deep-learning", proficiency: 85 },
  { id: 9, name: "Transformers", category: "deep-learning", proficiency: 88 },
  { id: 10, name: "Graph Neural Networks", category: "deep-learning", proficiency: 75 },
  { id: 11, name: "OpenAI Gym", category: "deep-learning", proficiency: 80 },
  
  // Data Processing
  { id: 12, name: "SQL", category: "data-processing", proficiency: 90 },
  { id: 13, name: "PostgreSQL", category: "data-processing", proficiency: 85 },
  { id: 14, name: "MySQL", category: "data-processing", proficiency: 80 },
  { id: 15, name: "MATLAB", category: "data-processing", proficiency: 75 },
  
  // Cloud Computing
  { id: 16, name: "AWS", category: "cloud-computing", proficiency: 90 },
  { id: 17, name: "Azure", category: "cloud-computing", proficiency: 85 },
  { id: 18, name: "GCP", category: "cloud-computing", proficiency: 80 },
  { id: 19, name: "Docker", category: "cloud-computing", proficiency: 85 },
  { id: 20, name: "Kubernetes", category: "cloud-computing", proficiency: 80 },
  { id: 21, name: "Terraform", category: "cloud-computing", proficiency: 75 },
  
  // Programming
  { id: 22, name: "Python", category: "programming", proficiency: 95 },
  { id: 23, name: "C++", category: "programming", proficiency: 80 },
  { id: 24, name: "Java", category: "programming", proficiency: 70 },
  { id: 25, name: "JavaScript", category: "programming", proficiency: 88 },
  { id: 26, name: "React", category: "programming", proficiency: 85 },
  { id: 27, name: "TypeScript", category: "programming", proficiency: 80 },
  { id: 28, name: "Node.js", category: "programming", proficiency: 75 },
  
  // Tools & Libraries
  { id: 29, name: "Git", category: "tools", proficiency: 90 },
  { id: 30, name: "Jupyter", category: "tools", proficiency: 95 },
  { id: 31, name: "HTML5", category: "tools", proficiency: 90 },
  { id: 32, name: "CSS3", category: "tools", proficiency: 85 }
];

export const contactInfo: ContactInfo = {
  email: "musalojidhiren@gmail.com",
  linkedin: "https://www.linkedin.com/in/musalojidhiren/",
  github: "https://github.com/SaiDhiren-Musaloji",
  phone: "+1 (862) 423-8830"
};

export const personalInfo = {
  name: "Sai Dhiren Musaloji",
  title: "Data Scientist & AI Research Engineer",
  location: "New Jersey, USA",
  education: "MS in Data Science - New Jersey Institute of Technology (GPA: 3.85/4.0)",
  experience: "AI Research Intern at Tech Mahindra, Cloud Computing Intern at LTIMindtree",
  about: "Passionate data scientist and AI research engineer with expertise in machine learning, deep learning, and cloud computing. Recent graduate with MS in Data Science from NJIT with a strong focus on transformer models, reinforcement learning, and ethical AI practices. Experienced in developing scalable solutions and contributing to national AI initiatives.",
  aboutDetailed: "My journey into data science began with a curiosity about how machines can learn and make intelligent decisions. Coming from an Electronics and Communication Engineering background, I discovered the fascinating intersection of mathematics, programming, and human cognition that defines modern AI.\n\nWhat drives me is the potential of AI to solve real-world problems that impact people's lives. During my time at Tech Mahindra's Makers Lab, I contributed to Project Indus, a national initiative to develop Large Language Models for Indian languages.\n\nI believe in the power of continuous learning and experimentation. My portfolio reflects this philosophy - from implementing adversarial attacks to understand AI vulnerabilities, to building RAG systems for intelligent research assistance."
};

export const experience = [
  {
    id: 1,
    title: "AI Research Intern",
    company: "Tech Mahindra, Makers Lab",
    location: "Pune, India",
    duration: "Oct 2023 - Jan 2024",
    description: [
      "Contributed to Project Indus, a national initiative to develop a Large Language Model supporting over 15 Indian languages",
      "Developed and optimized data preprocessing pipelines for tokenization, normalization, and character embedding in low-resource dialects",
      "Benchmarked transformer models using metrics like perplexity, BLEU, and F1 to evaluate performance on complex language tasks",
      "Collaborated with cross-functional teams to promote ethical AI practices and model alignment for underrepresented communities"
    ]
  },
  {
    id: 2,
    title: "Cloud Computing Intern",
    company: "LTIMindtree",
    location: "Hyderabad, Telangana, India",
    duration: "Feb 2023 - Apr 2023",
    description: [
      "Gained comprehensive knowledge of cloud computing fundamentals, focusing on Amazon Web Services (AWS)",
      "Accelerated data processing workflows using EC2, S3, Lambda, VPC, and RDS, alongside Docker, Kubernetes, and Terraform",
      "Participated in cloud migration and big data processing initiatives, optimizing resource allocation and cost-efficiency",
      "Developed hands-on expertise in infrastructure-as-code (IaC), container orchestration, and DevOps best practices"
    ]
  }
];

export const education = [
  {
    id: 1,
    degree: "MS in Data Science",
    school: "New Jersey Institute of Technology",
    duration: "Jan 2024 - May 2025",
    gpa: "3.85/4.0",
    coursework: [
      "Machine Learning",
      "Deep Learning", 
      "Machine Learning for Time Series Analysis",
      "Cloud Computing",
      "Artificial Intelligence",
      "Database Management Systems and Design",
      "Big Data",
      "Applied Statistics",
      "Python and Mathematics"
    ]
  },
  {
    id: 2,
    degree: "B.Tech in Electronics and Communication Engineering",
    school: "Mahatma Gandhi Institute of Technology",
    duration: "Jun 2019 - Apr 2023",
    gpa: "3.0/4.0",
    capstone: "Designed and implemented a Handwritten Telugu Character Recognition framework utilizing Optical Character Recognition (OCR) methods integrated with Convolutional Neural Networks (CNNs) in TensorFlow and Keras. Led data acquisition and preprocessing, constructing a diverse, high-dimensional dataset representing handwriting variability across users. Trained and evaluated deep learning models for character-level recognition with robust generalization."
  }
];

export const projectCategories = [
  { id: "all", name: "All Projects", count: 17 },
  { id: "data-analysis", name: "Data Analysis", count: 3 },
  { id: "data-engineering", name: "Data Engineering", count: 1 },
  { id: "ml-modeling", name: "ML Modeling", count: 7 },
  { id: "engineering-solutions", name: "Engineering Solutions", count: 3 },
  { id: "generative-ai", name: "Generative AI", count: 3 }
]; 