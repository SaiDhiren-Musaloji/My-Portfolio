import { Project, Certificate, Skill, ContactInfo } from '../types';

export const projects: Project[] = [
  {
    id: 1,
    title: "AI Research Assistant with Retrieval-Augmented Generation",
    description: "• Traditional research assistants lack contextual understanding and fail to provide accurate, up-to-date information from multiple sources with proper citation and verification\n\n• Developed intelligent research assistant integrating Google Gemini 1.5 Flash API with RAG, building custom vector database using scikit-learn's NearestNeighbors, implementing newspaper3k for article extraction, and engineering end-to-end workflow with multi-model NLP processing (BART, BERT, DistilBERT) for contextually-grounded responses.",
    image: "/images/projects/Rag.png",
    technologies: ["Python", "Google Gemini API", "BART", "BERT", "DistilBERT", "RAG", "NLP", "newspaper3k", "scikit-learn", "Vector Database"],
    githubUrl: "https://github.com/SaiDhiren-Musaloji?tab=repositories",
    category: "Generative AI",
    primaryCategory: "generative-ai",
    categories: ["generative-ai", "nlp-speech", "cloud-mlops"],
    detailedDescription: "Developed an intelligent research assistant integrating Google Gemini 1.5 Flash API with retrieval-augmented generation, combining web search capabilities with contextual document processing. Built custom data processing pipeline integrating newspaper3k for article extraction, serper.dev for search results, with custom error handling, rate limiting, and multi-model NLP processing (BART, BERT, DistilBERT). Implemented custom vector database solution from scratch using scikit-learn's NearestNeighbors and numpy arrays for in-memory semantic similarity search and document retrieval. Engineered end-to-end RAG workflow orchestrating pre-trained transformer models for embedding generation, custom retrieval logic, and dynamic prompt construction for contextually-grounded responses. Designed modular architecture with asynchronous processing, session management, and multi-model integration for summarization, named entity recognition, and sentiment analysis."
  },
  {
    id: 3,
    title: "Adversarial Attack on CIFAR-10 Models",
    description: "• Minor, imperceptible perturbations to input images can fool state-of-the-art deep learning models into making incorrect predictions, highlighting critical security vulnerabilities in AI systems\n\n• Implemented ensemble-based adversarial attacks using Iterative Fast Gradient Sign Method (I-FGSM) with gradient aggregation across multiple pre-trained models, achieving 85% attack success rate while maintaining visual imperceptibility.",
    image: "/images/projects/Adversarial.png",
    technologies: ["Python", "PyTorch", "TensorFlow", "CIFAR-10", "Adversarial ML", "Computer Vision"],
    githubUrl: "https://github.com/SaiDhiren-Musaloji?tab=repositories",
    category: "ML Modeling",
    primaryCategory: "machine-learning",
    categories: ["machine-learning"],
    detailedDescription: "Implemented ensemble-based adversarial attacks using Iterative Fast Gradient Sign Method (I-FGSM) with gradient aggregation across multiple pre-trained models, achieving 85% attack success rate while maintaining visual imperceptibility."
  },
  {
    id: 4,
    title: "BERT-Based Reading Comprehension",
    description: "• Traditional question answering systems struggle to understand contextual relationships and accurately extract answer spans from given passages\n\n• Fine-tuned BERT model with custom loss functions for start/end position prediction, implementing dynamic learning rate scheduling and gradient accumulation techniques for improved span extraction accuracy.",
    image: "/images/projects/bert.png",
    technologies: ["Python", "BERT", "Transformers", "PyTorch", "NLP", "Question Answering"],
    githubUrl: "https://github.com/SaiDhiren-Musaloji?tab=repositories",
    category: "Generative AI",
    primaryCategory: "generative-ai",
    categories: ["generative-ai", "nlp-speech"]
  },
  {
    id: 5,
    title: "Deep Reinforcement Learning for Lunar Lander",
    description: "• Learning complex sequential decision-making under uncertainty where an agent must balance multiple competing objectives including safe landing, fuel efficiency, and trajectory optimization in continuous action spaces\n\n• Implemented Actor-Critic architecture with policy gradient methods, custom reward shaping, and experience replay mechanisms achieving consistent successful landings with scores above 200 points.",
    image: "/images/projects/lunarlander.png",
    technologies: ["Python", "PyTorch", "OpenAI Gym", "Reinforcement Learning", "Actor-Critic"],
    githubUrl: "https://github.com/SaiDhiren-Musaloji?tab=repositories",
    category: "ML Modeling",
    primaryCategory: "machine-learning",
    categories: ["machine-learning"]
  },
  {
    id: 6,
    title: "Graph Neural Networks for Financial Prediction",
    description: "• Traditional time series models fail to capture complex inter-asset relationships and network effects that significantly influence financial market dynamics\n\n• Implemented Graph Attention Networks (GAT) to model both spatial relationships between assets and temporal dependencies, achieving superior prediction accuracy through ensemble strategies and multi-head attention mechanisms.",
    image: "/images/projects/deep_learning.png",
    technologies: ["Python", "PyTorch", "Graph Neural Networks", "GAT", "Financial ML"],
    githubUrl: "https://github.com/SaiDhiren-Musaloji?tab=repositories",
    category: "ML Modeling",
    primaryCategory: "machine-learning",
    categories: ["machine-learning"]
  },
  {
    id: 7,
    title: "Transformer-Based Speaker Classification",
    description: "• Traditional RNN-based approaches struggle to capture long-range dependencies in audio sequences and fail to effectively classify speakers from variable-length MFCC features across 600 different speakers\n\n• Implemented transformer encoder architecture with custom positional encoding for audio data, achieving superior accuracy through multi-head self-attention mechanisms and mixed precision training.",
    image: "/images/projects/Transformer-Based Speaker Classification.png",
    technologies: ["Python", "PyTorch", "Transformers", "Audio Processing", "MFCC"],
    githubUrl: "https://github.com/SaiDhiren-Musaloji?tab=repositories",
    category: "ML Modeling",
    primaryCategory: "nlp-speech",
    categories: ["nlp-speech", "machine-learning"]
  },
  {
    id: 8,
    title: "Self-Supervised Learning for User Localization",
    description: "• Traditional wireless localization methods require extensive labeled data and fail to extract meaningful representations from complex channel state information for accurate 3D position prediction\n\n• Implemented two-stage self-supervised approach using autoencoder for feature extraction followed by position prediction model, leveraging unlabeled channel data to achieve accurate localization with limited labeled samples.",
    image: "/images/projects/Self-Supervised Learning for User Localization.jpg",
    technologies: ["Python", "PyTorch", "Self-Supervised Learning", "Autoencoders", "Wireless ML"],
    githubUrl: "https://github.com/SaiDhiren-Musaloji?tab=repositories",
    category: "ML Modeling",
    primaryCategory: "machine-learning",
    categories: ["machine-learning"]
  },
  {
    id: 9,
    title: "Neural Machine Translation with Transformers",
    description: "• Traditional RNN-based translation systems struggle with long-range dependencies, parallel processing limitations, and maintaining contextual accuracy across variable-length sequences in English to Traditional Chinese translation\n\n• Implemented transformer encoder-decoder architecture with SentencePiece tokenization, custom beam search decoding, and label smoothing regularization achieving superior translation quality and reduced training time.",
    image: "/images/projects/bert.png",
    technologies: ["Python", "PyTorch", "Transformers", "SentencePiece", "NMT"],
    githubUrl: "https://github.com/SaiDhiren-Musaloji?tab=repositories",
    category: "Generative AI",
    primaryCategory: "generative-ai",
    categories: ["generative-ai", "nlp-speech"]
  },
  {
    id: 10,
    title: "Deep Neural Networks for Speech Classification",
    description: "• Traditional machine learning approaches fail to capture complex non-linear relationships in acoustic feature space and struggle with inherent variability in speech patterns across different speakers and contexts for phoneme classification\n\n• Implemented multi-layer neural network with context window concatenation, batch normalization, and advanced data augmentation techniques achieving substantial accuracy improvements over baseline methods.",
    image: "/images/projects/Deep Neural Networks for Speech Classification.webp",
    technologies: ["Python", "PyTorch", "Speech Processing", "MFCC", "Phoneme Classification"],
    githubUrl: "https://github.com/SaiDhiren-Musaloji?tab=repositories",
    category: "ML Modeling",
    primaryCategory: "nlp-speech",
    categories: ["nlp-speech", "machine-learning"]
  },
  {
    id: 11,
    title: "Handwritten Telugu Character Recognition",
    description: "• Traditional OCR systems struggle with complex Indic scripts like Telugu due to character variations, ligatures, and contextual dependencies\n\n• Implemented CNN-based OCR framework with comprehensive data preprocessing, custom character segmentation, and multi-scale feature extraction achieving robust recognition accuracy for Telugu script.",
    image: "/images/projects/telugu handwritten recognition.png",
    technologies: ["TensorFlow", "Keras", "CNN", "OpenCV", "OCR"],
    githubUrl: "https://github.com/SaiDhiren-Musaloji?tab=repositories",
    category: "ML Modeling",
    primaryCategory: "nlp-speech",
    categories: ["nlp-speech", "machine-learning"]
  },
  {
    id: 12,
    title: "Distributed Recognition Engine Using Cloud-Native Paradigms",
    description: "Problem: Traditional image and text recognition systems lack scalability and reliability for high-volume processing with varying workloads. Solution: Built asynchronous recognition pipeline using AWS EC2, SQS, S3, and Rekognition with stateless compute strategies, achieving 99.9% uptime through visibility timeout calibration and retry mechanisms.",
    image: "/images/projects/sqs.png",
    technologies: ["AWS", "Docker", "Kubernetes", "Terraform", "Cloud Computing"],
    githubUrl: "https://github.com/SaiDhiren-Musaloji?tab=repositories",
    category: "Engineering Solutions",
    primaryCategory: "systems-design",
    categories: ["systems-design", "cloud-mlops", "data-engineering"]
  },
  {
    id: 13,
    title: "Spatiotemporal Forecasting of Urban Traffic Networks",
    description: "Problem: Traditional traffic prediction models fail to capture complex spatiotemporal dependencies and topological interactions between network nodes in urban traffic systems. Solution: Engineered fusion architecture combining ARIMA, LSTM, and GNN models with PCA and t-SNE dimensionality reduction, achieving robust multi-horizon prediction across 325 sensors in the PEMS-BAY dataset.",
    image: "/images/projects/traffic_forecasting.png",
    technologies: ["Python", "ARIMA", "LSTM", "GNN", "PCA", "t-SNE", "Time Series"],
    githubUrl: "https://github.com/SaiDhiren-Musaloji?tab=repositories",
    category: "Data Analysis",
    primaryCategory: "analytics-bi",
    categories: ["analytics-bi", "machine-learning"]
  },
  {
    id: 14,
    title: "Parallelized ML Pipeline for Oenological Forecasting",
    description: "Problem: Traditional wine quality prediction models lack scalability and fail to handle large-scale datasets efficiently, resulting in slow training times and limited throughput for production inference. Solution: Created Spark-based distributed system on AWS EMR with partitioned data and in-memory caching, achieving 60% training speed improvement and high-throughput inference with autoscaling and performance logging.",
    image: "/images/projects/Parallel.png",
    technologies: ["Apache Spark", "AWS EMR", "ML Pipeline", "Distributed Computing", "Wine Quality Prediction"],
    githubUrl: "https://github.com/SaiDhiren-Musaloji?tab=repositories",
    category: "Engineering Solutions",
    primaryCategory: "data-engineering",
    categories: ["data-engineering", "cloud-mlops", "systems-design"]
  },
  {
    id: 15,
    title: "Flight Data Analysis - Big Data Analytics & Performance Optimization",
    description: "Engineered scalable AWS-based MapReduce data processing pipelines using Apache Spark and Oozie orchestration, analyzing 22-year aviation dataset (1987-2008) containing 120M+ flight records with 40% improved query performance optimization. Implemented comprehensive scalability testing framework across varying data volumes (1GB to 500GB), demonstrating linear performance scalability and optimizing AWS resource allocation to reduce processing costs by 35%. Established high-performance aviation analytics platform supporting 15 concurrent users with sub-3-second query response times, delivering actionable insights on 5,000+ flight routes for executive decision-making processes.",
    image: "/images/projects/flight_data.png",
    technologies: ["AWS", "Apache Spark", "MapReduce", "Apache Oozie", "Big Data", "Data Analysis", "Scalability Testing", "Performance Optimization"],
    githubUrl: "https://github.com/SaiDhiren-Musaloji?tab=repositories",
    category: "Data Analysis",
    primaryCategory: "analytics-bi",
    categories: ["analytics-bi", "data-engineering", "cloud-mlops"]
  },
  {
    id: 16,
    title: "Equity Portfolio Management - Financial Analytics & Quantitative Analysis",
    description: "Orchestrated comprehensive portfolio optimization analysis using Python (Pandas, NumPy, Matplotlib) to evaluate $5M equity portfolio, implementing advanced risk-return modeling that generated 18% annual return improvement over S&P 500 benchmark. Directed quantitative analysis team of 3 members in developing algorithmic position sizing strategies, utilizing simulations across 500+ securities to reduce portfolio volatility by 23% while maintaining 12.4% target return threshold. Architected automated performance tracking system monitoring 15 key performance indicators (KPIs) daily, enabling real-time investment decisions that improved Sharpe ratio from 1.2 to 1.8 over 12-month analytical period.",
    image: "/images/projects/portfolio.png",
    technologies: ["Python", "Pandas", "NumPy", "Matplotlib", "Quantitative Analysis", "Risk Modeling", "Algorithmic Trading", "Portfolio Optimization", "Sharpe Ratio"],
    githubUrl: "https://github.com/SaiDhiren-Musaloji?tab=repositories",
    category: "Engineering Solutions",
    primaryCategory: "analytics-bi",
    categories: ["analytics-bi"]
  },
  {
    id: 17,
    title: "Banking Transaction Management System - Agile Full-Stack RDBMS Development & Analytics",
    description: "Engineered enterprise-grade relational database management system integrating MySQL primary databases with Snowflake data warehouse and DynamoDB document storage for banking network infrastructure, designing normalized schema with 15+ tables to handle 10,000+ daily transactions with 99.9% data integrity and sub-200ms query response times. Developed user-centric web interface using Flask and REST APIs to support comprehensive transaction processing workflows, implementing real-time balance updates and transaction history tracking for 500+ concurrent users based on stakeholder requirements. Implemented agile development methodology across 2-sprint delivery cycle, conducting user story analysis and stakeholder requirement gathering to deliver full-stack banking solution with transaction management, user authentication, and reporting capabilities.",
    image: "/images/projects/database.png",
    technologies: ["MySQL", "Snowflake", "DynamoDB", "Flask", "REST APIs", "Python", "Agile Development", "Database Design", "Web Interface", "Transaction Management"],
    githubUrl: "https://github.com/SaiDhiren-Musaloji?tab=repositories",
    category: "Data Engineering",
    primaryCategory: "systems-design",
    categories: ["systems-design", "data-engineering"]
  },
  {
    id: 18,
    title: "Real-Time Ride-Matching Platform (Uber-Like System)",
    description: "Architected a dual-index geospatial platform processing 50K+ GPS updates/min; Redis Geo for low-latency lookups (80 ms p50) and PostGIS for durable analytics, yielding a 68% reduction in match latency. Built a streaming ingestion pipeline with coordinate normalization and dual-write patterns; achieved sub-1s end-to-end data freshness and 99.9% location accuracy in load tests. Implemented distributed locking and idempotent booking flows (Redis) to guarantee exactly-one driver assignment; eliminated duplicate rides across 5K+ concurrent booking requests. Engineered WebSocket fanout delivering ETAs and booking state to 5K+ concurrent clients with ~200 ms median propagation; optimized connection pooling and broadcast paths. Designed an event-driven, queue-based architecture decoupling booking, availability, and notifications with backpressure, exponential backoff, and dead-letter queues for resilience under surges. Optimized hot paths with TTL-based cache eviction, selective invalidation, and time-partitioned storage; reduced DB load ~60% while maintaining 85%+ cache hit rates. Built end-to-end observability: match latency, lock contention, throughput, and data freshness; added reconciliation to detect Redis↔Postgres drift with automated consistency checks and structured logs. Containerized services with Docker and automated synthetic city-scale benchmarks and capacity planning to validate SLOs and scaling thresholds.",
    image: "/images/projects/placeholder.png",
    technologies: ["Redis Geo", "PostgreSQL", "PostGIS", "WebSockets", "Node.js", "Python", "Docker", "Distributed Systems", "Geospatial Indexing", "Queue-based Architecture", "Monitoring", "Observability"],
    githubUrl: "https://github.com/SaiDhiren-Musaloji?tab=repositories",
    category: "Engineering Solutions",
    primaryCategory: "systems-design",
    categories: ["systems-design", "data-engineering", "cloud-mlops"],
    detailedDescription: "Real-Time Ride-Matching Platform (Uber-Like System) - Architected a dual-index geospatial platform processing 50K+ GPS updates/min; Redis Geo for low-latency lookups (80 ms p50) and PostGIS for durable analytics, yielding a 68% reduction in match latency. Built a streaming ingestion pipeline with coordinate normalization and dual-write patterns; achieved sub-1s end-to-end data freshness and 99.9% location accuracy in load tests. Implemented distributed locking and idempotent booking flows (Redis) to guarantee exactly-one driver assignment; eliminated duplicate rides across 5K+ concurrent booking requests. Engineered WebSocket fanout delivering ETAs and booking state to 5K+ concurrent clients with ~200 ms median propagation; optimized connection pooling and broadcast paths. Designed an event-driven, queue-based architecture decoupling booking, availability, and notifications with backpressure, exponential backoff, and dead-letter queues for resilience under surges. Optimized hot paths with TTL-based cache eviction, selective invalidation, and time-partitioned storage; reduced DB load ~60% while maintaining 85%+ cache hit rates. Built end-to-end observability: match latency, lock contention, throughput, and data freshness; added reconciliation to detect Redis↔Postgres drift with automated consistency checks and structured logs. Containerized services with Docker and automated synthetic city-scale benchmarks and capacity planning to validate SLOs and scaling thresholds. Impact: Zero duplicate bookings in stress tests; sustained 50K+ GPS updates/min; ~60% DB load reduction; sub-200 ms client updates at 5K+ concurrency; ~99.9% uptime in test environments."
  }

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
  // Machine Learning & Deep Learning
  { id: 1, name: "Scikit-learn", category: "machine-learning", proficiency: 88 },
  { id: 2, name: "Pandas", category: "machine-learning", proficiency: 95 },
  { id: 3, name: "NumPy", category: "machine-learning", proficiency: 92 },
  { id: 4, name: "Power BI", category: "machine-learning", proficiency: 85 },
  { id: 5, name: "R", category: "machine-learning", proficiency: 75 },
  { id: 6, name: "XGBoost", category: "machine-learning", proficiency: 80 },
  { id: 7, name: "Ensemble Methods", category: "machine-learning", proficiency: 85 },
  
  // Deep Learning
  { id: 8, name: "TensorFlow", category: "deep-learning", proficiency: 90 },
  { id: 9, name: "Keras", category: "deep-learning", proficiency: 88 },
  { id: 10, name: "PyTorch", category: "deep-learning", proficiency: 92 },
  { id: 11, name: "BERT", category: "deep-learning", proficiency: 85 },
  { id: 12, name: "Transformers", category: "deep-learning", proficiency: 88 },
  { id: 13, name: "LSTM", category: "deep-learning", proficiency: 85 },
  { id: 14, name: "CNN", category: "deep-learning", proficiency: 90 },
  { id: 15, name: "Graph Neural Networks", category: "deep-learning", proficiency: 75 },
  { id: 16, name: "Reinforcement Learning", category: "deep-learning", proficiency: 80 },
  { id: 17, name: "GANs", category: "deep-learning", proficiency: 75 },
  { id: 18, name: "OpenAI Gym", category: "deep-learning", proficiency: 80 },
  
  // Data Processing & Analytics
  { id: 19, name: "SQL", category: "data-processing", proficiency: 90 },
  { id: 20, name: "PostgreSQL", category: "data-processing", proficiency: 85 },
  { id: 21, name: "MySQL", category: "data-processing", proficiency: 80 },
  { id: 22, name: "MongoDB", category: "data-processing", proficiency: 75 },
  { id: 23, name: "Snowflake", category: "data-processing", proficiency: 80 },
  { id: 24, name: "DynamoDB", category: "data-processing", proficiency: 75 },
  { id: 25, name: "Data Warehousing", category: "data-processing", proficiency: 80 },
  { id: 26, name: "ETL/ELT", category: "data-processing", proficiency: 85 },
  { id: 27, name: "Statistics", category: "data-processing", proficiency: 85 },
  { id: 28, name: "Time Series Forecasting", category: "data-processing", proficiency: 80 },
  { id: 29, name: "PCA", category: "data-processing", proficiency: 80 },
  { id: 30, name: "MATLAB", category: "data-processing", proficiency: 75 },
  
  // Big Data & Distributed Systems
  { id: 31, name: "Apache Spark", category: "data-processing", proficiency: 85 },
  { id: 32, name: "Hadoop", category: "data-processing", proficiency: 75 },
  { id: 33, name: "Kafka", category: "data-processing", proficiency: 70 },
  { id: 34, name: "AWS EMR", category: "cloud-computing", proficiency: 80 },
  { id: 35, name: "MapReduce", category: "data-processing", proficiency: 75 },
  { id: 36, name: "Distributed Computing", category: "data-processing", proficiency: 80 },
  { id: 37, name: "Apache Airflow", category: "tools", proficiency: 75 },
  
  // Cloud Computing
  { id: 38, name: "AWS", category: "cloud-computing", proficiency: 90 },
  { id: 39, name: "Azure", category: "cloud-computing", proficiency: 85 },
  { id: 40, name: "GCP", category: "cloud-computing", proficiency: 80 },
  { id: 41, name: "Docker", category: "cloud-computing", proficiency: 85 },
  { id: 42, name: "Kubernetes", category: "cloud-computing", proficiency: 80 },
  { id: 43, name: "Terraform", category: "cloud-computing", proficiency: 75 },
  { id: 44, name: "SageMaker", category: "cloud-computing", proficiency: 80 },
  { id: 45, name: "Redshift", category: "cloud-computing", proficiency: 75 },
  
  // Programming
  { id: 46, name: "Python", category: "programming", proficiency: 95 },
  { id: 47, name: "R", category: "programming", proficiency: 75 },
  { id: 48, name: "JavaScript", category: "programming", proficiency: 88 },
  { id: 49, name: "TypeScript", category: "programming", proficiency: 80 },
  { id: 50, name: "React", category: "programming", proficiency: 85 },
  { id: 51, name: "Java", category: "programming", proficiency: 70 },
  { id: 52, name: "Flask", category: "programming", proficiency: 80 },
  { id: 53, name: "REST APIs", category: "programming", proficiency: 85 },
  
  // Natural Language Processing
  { id: 54, name: "NLTK", category: "machine-learning", proficiency: 80 },
  { id: 55, name: "spaCy", category: "machine-learning", proficiency: 80 },
  { id: 56, name: "RAG", category: "deep-learning", proficiency: 85 },
  { id: 57, name: "Vector Databases", category: "data-processing", proficiency: 75 },
  { id: 58, name: "Text Classification", category: "machine-learning", proficiency: 85 },
  
  // Data Visualization
  { id: 59, name: "Tableau", category: "tools", proficiency: 80 },
  { id: 60, name: "Plotly", category: "tools", proficiency: 75 },
  { id: 61, name: "Interactive Dashboards", category: "tools", proficiency: 80 },
  { id: 62, name: "Business Intelligence", category: "tools", proficiency: 80 },
  
  // Tools & Libraries
  { id: 63, name: "Git", category: "tools", proficiency: 90 },
  { id: 64, name: "Jupyter", category: "tools", proficiency: 95 },
  { id: 65, name: "HTML5", category: "tools", proficiency: 90 },
  { id: 66, name: "CSS3", category: "tools", proficiency: 85 }
];

export const contactInfo: ContactInfo = {
  email: "musalojidhiren@gmail.com",
  linkedin: "https://www.linkedin.com/in/musalojidhiren/",
  github: "https://github.com/SaiDhiren-Musaloji",
  phone: "+1 (862) 423-8830"
};

export const personalInfo = {
  name: "Sai Dhiren Musaloji",
  title: "Data Engineer & AI Engineer",
  location: "Dallas, TX",
  education: "MS in Data Science - New Jersey Institute of Technology (GPA: 3.85/4.0)",
  experience: "Data Engineer at TAWIN Solutions LLC",
  about: "Data Engineer and AI Engineer with expertise in statistical analysis, machine learning, and cloud computing. Currently building data-driven recommendation systems and analytics platforms at TAWIN Solutions. MS in Data Science from NJIT with hands-on experience in multilingual LLM research, real-time data pipelines, and enterprise analytics solutions.",
  aboutDetailed: "I'm a passionate Data Engineer and AI Engineer with a Master's in Data Science from New Jersey Institute of Technology, where I graduated with a 3.85 GPA. My journey in data science combines rigorous academic training with hands-on industry experience, enabling me to bridge the gap between complex analytical problems and practical, impactful solutions.\n\nCurrently, I'm working as a Data Engineer at TAWIN Solutions LLC in Dallas, where I lead statistical analysis on user preference patterns and build comprehensive data pipelines that power personalized recommendation systems. My work involves integrating real-time APIs, performing sentiment analysis, and developing interactive dashboards that drive data-driven decision-making. I've successfully improved recommendation click-through rates by 23% and achieved 99.2% data accuracy across integrated sources.\n\nPreviously, I contributed to cutting-edge AI research at Tech Mahindra's Makers Lab, where I managed data pipelines for a major multilingual LLM research project spanning 15+ languages. I engineered real-time transcription systems, prototyped semantic search architectures, and built full-stack data flows for VR research platforms. This experience gave me deep exposure to production-grade MLOps and enterprise NLP system design.\n\nMy technical expertise spans machine learning, deep learning, cloud computing, and data engineering. I'm proficient in Python, SQL, Azure, Power BI, and various ML frameworks. I've built everything from RAG-powered research assistants to adversarial attack systems, from transformer-based models to distributed data processing pipelines.\n\nWhat sets me apart is my ability to translate complex analytical insights into actionable business strategies. I thrive in collaborative environments where I can work with cross-functional teams to solve challenging problems. Whether it's optimizing recommendation algorithms, building predictive models, or architecting scalable data solutions, I'm driven by the impact that data-driven decisions can have on business outcomes and user experiences.\n\nI'm always eager to take on new challenges and continue learning. My portfolio reflects my commitment to continuous growth—from understanding AI vulnerabilities through adversarial attacks to building enterprise-grade analytics platforms. I believe in the power of data to transform businesses and improve lives, and I'm excited to contribute to innovative projects that push the boundaries of what's possible with AI and analytics."
};

export const experience = [
  {
    id: 1,
    title: "Data Engineer",
    company: "TAWIN Solutions",
    location: "Dallas, TX",
    duration: "May 2025 – Present",
    description: [
      "Designed, implemented, and optimized scalable ETL pipelines using SSIS and Python to facilitate the transformation and linkage of data across multiple sources, automating data workflows and establishing data validation frameworks.",
      "Developed and optimized complex SQL queries and stored procedures in MS SQL Server (Azure and On-Prem), leveraging Window Functions and Aggregate Functions for efficient data joining, filtering, and aggregation.",
      "Managed data quality and standardization efforts by conducting code reviews and ensuring 99% data accuracy across enterprise data warehouse solutions.",
      "Translated business requirements into technical BI solutions, designing robust data models supporting analytics and executive reporting needs via Power BI dashboards.",
      "Environment: MS SQL Server, Azure, SSIS, Power BI, Data Warehousing, ETL, DAX."
    ]
  },
  {
    id: 2,
    title: "AI Research Data Analyst Intern",
    company: "Tech Mahindra, Makers Lab R&D",
    location: "Pune, India",
    duration: "Oct 2023 – Jan 2024",
    description: [
      "Developed pipeline architecture for multilingual LLM research across multiple native languages, establishing preprocessing standards and quality benchmarks for production-grade Natural Language Processing (NLP) systems.",
      "Architected real-time German-to-English transcription system integrating commercial APIs with custom processing layer, achieving significant improvements in contextual understanding for internal research use cases.",
      "Prototyped semantic search architecture deploying vector-based mechanisms and structured full-stack workflow with REST API integration for a VR research platform.",
      "Environment: Python, Transformers, Natural Language Processing (NLP), REST APIs, Data Preprocessing, Annotation Tools, API Integration."
    ]
  },
  {
    id: 3,
    title: "Data Analyst",
    company: "Logisoft Technologies",
    location: "Hyderabad, India",
    duration: "Dec 2022 – Oct 2023",
    description: [
      "Deployed robust Extract, Transform, Load (ETL) processes using Apache Airflow across Snowflake, DynamoDB, and SQL Server, automating workflows and reducing manual processing time by 75%.",
      "Orchestrated enterprise-wide data consolidation of 50,000+ records, building a predictive utilization model that reduced unnecessary spending by 18%.",
      "Led cross-functional analytics initiatives between business units and technical teams, translating requirements into solutions and presenting data insights to C-level executives.",
      "Environment: Power BI, SQL, Python, Apache Airflow, Snowflake, DynamoDB."
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
    duration: "Jun 2019 - Jun 2023",
    gpa: "3.00/4.0",
    capstone: "Designed and implemented a Handwritten Telugu Character Recognition framework utilizing Optical Character Recognition (OCR) methods integrated with Convolutional Neural Networks (CNNs) in TensorFlow and Keras. Led data acquisition and preprocessing, constructing a diverse, high-dimensional dataset representing handwriting variability across users. Trained and evaluated deep learning models for character-level recognition with robust generalization."
  }
];

export const projectCategories = [
  { id: "all", name: "All Projects", count: 17 },
  { id: "generative-ai", name: "Generative AI", count: 3 },
  { id: "nlp-speech", name: "NLP & Speech", count: 5 },
  { id: "machine-learning", name: "Machine Learning", count: 8 },
  { id: "data-engineering", name: "Data Engineering", count: 6 },
  { id: "systems-design", name: "Systems Design", count: 5 },
  { id: "cloud-mlops", name: "Cloud & MLOps", count: 6 },
  { id: "analytics-bi", name: "Analytics & BI", count: 3 }
];