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
    detailedDescription: "Developed an intelligent research assistant integrating Google Gemini 1.5 Flash API with retrieval-augmented generation, combining web search capabilities with contextual document processing. Built custom data processing pipeline integrating newspaper3k for article extraction, serper.dev for search results, with custom error handling, rate limiting, and multi-model NLP processing (BART, BERT, DistilBERT). Implemented custom vector database solution from scratch using scikit-learn's NearestNeighbors and numpy arrays for in-memory semantic similarity search and document retrieval. Engineered end-to-end RAG workflow orchestrating pre-trained transformer models for embedding generation, custom retrieval logic, and dynamic prompt construction for contextually-grounded responses. Designed modular architecture with asynchronous processing, session management, and multi-model integration for summarization, named entity recognition, and sentiment analysis."
  },
  {
    id: 2,
    title: "Software License Compliance Analysis - Enterprise Data Analytics",
    description: "Orchestrated multi-source data integration project across 25 enterprise systems including Snowflake data warehouse, DynamoDB NoSQL clusters, SQL Server databases, and Cassandra distributed systems, implementing robust ETL processes using Apache Airflow to consolidate 500K+ software license records with 99.5% accuracy for regulatory compliance reporting. Spearheaded development of comprehensive compliance analytics platform using Python and SQL, reducing manual audit processing time by 60% and identifying $180K in cost-optimization opportunities through intelligent license allocation strategies. Constructed interactive Power BI dashboards serving 45+ stakeholders with real-time monitoring capabilities for 1,200+ software licenses, implementing automated alert systems for free, enterprise, and client-billed license categories. Executed advanced pattern recognition analysis on 3-year historical licensing data using machine learning algorithms, achieving 94% accuracy in utilization trend prediction and preventing $75K in unnecessary license renewals.",
    image: "/images/projects/SC.webp",
    technologies: ["Python", "SQL", "Power BI", "Apache Airflow", "Snowflake", "DynamoDB", "Machine Learning", "Data Analysis", "Cost Optimization", "ETL"],
    githubUrl: "https://github.com/SaiDhiren-Musaloji?tab=repositories",
    category: "Data Analysis",
    detailedDescription: "Built a comprehensive software license compliance and cost optimization system for enterprise environments. The solution analyzes software usage patterns, identifies compliance risks, and provides actionable insights for cost reduction."
  },
  {
    id: 3,
    title: "Adversarial Attack on CIFAR-10 Models",
    description: "• Minor, imperceptible perturbations to input images can fool state-of-the-art deep learning models into making incorrect predictions, highlighting critical security vulnerabilities in AI systems\n\n• Implemented ensemble-based adversarial attacks using Iterative Fast Gradient Sign Method (I-FGSM) with gradient aggregation across multiple pre-trained models, achieving 85% attack success rate while maintaining visual imperceptibility.",
    image: "/images/projects/Adversarial.png",
    technologies: ["Python", "PyTorch", "TensorFlow", "CIFAR-10", "Adversarial ML", "Computer Vision"],
    githubUrl: "https://github.com/SaiDhiren-Musaloji?tab=repositories",
    category: "ML Modeling",
    detailedDescription: "Implemented ensemble-based adversarial attacks using Iterative Fast Gradient Sign Method (I-FGSM) with gradient aggregation across multiple pre-trained models, achieving 85% attack success rate while maintaining visual imperceptibility."
  },
  {
    id: 4,
    title: "BERT-Based Reading Comprehension",
    description: "• Traditional question answering systems struggle to understand contextual relationships and accurately extract answer spans from given passages\n\n• Fine-tuned BERT model with custom loss functions for start/end position prediction, implementing dynamic learning rate scheduling and gradient accumulation techniques for improved span extraction accuracy.",
    image: "/images/projects/bert.png",
    technologies: ["Python", "BERT", "Transformers", "PyTorch", "NLP", "Question Answering"],
    githubUrl: "https://github.com/SaiDhiren-Musaloji?tab=repositories",
    category: "Generative AI"
  },
  {
    id: 5,
    title: "Deep Reinforcement Learning for Lunar Lander",
    description: "• Learning complex sequential decision-making under uncertainty where an agent must balance multiple competing objectives including safe landing, fuel efficiency, and trajectory optimization in continuous action spaces\n\n• Implemented Actor-Critic architecture with policy gradient methods, custom reward shaping, and experience replay mechanisms achieving consistent successful landings with scores above 200 points.",
    image: "/images/projects/lunarlander.png",
    technologies: ["Python", "PyTorch", "OpenAI Gym", "Reinforcement Learning", "Actor-Critic"],
    githubUrl: "https://github.com/SaiDhiren-Musaloji?tab=repositories",
    category: "ML Modeling"
  },
  {
    id: 6,
    title: "Graph Neural Networks for Financial Prediction",
    description: "• Traditional time series models fail to capture complex inter-asset relationships and network effects that significantly influence financial market dynamics\n\n• Implemented Graph Attention Networks (GAT) to model both spatial relationships between assets and temporal dependencies, achieving superior prediction accuracy through ensemble strategies and multi-head attention mechanisms.",
    image: "/images/projects/deep_learning.png",
    technologies: ["Python", "PyTorch", "Graph Neural Networks", "GAT", "Financial ML"],
    githubUrl: "https://github.com/SaiDhiren-Musaloji?tab=repositories",
    category: "ML Modeling"
  },
  {
    id: 7,
    title: "Transformer-Based Speaker Classification",
    description: "• Traditional RNN-based approaches struggle to capture long-range dependencies in audio sequences and fail to effectively classify speakers from variable-length MFCC features across 600 different speakers\n\n• Implemented transformer encoder architecture with custom positional encoding for audio data, achieving superior accuracy through multi-head self-attention mechanisms and mixed precision training.",
    image: "/images/projects/Transformer-Based Speaker Classification.png",
    technologies: ["Python", "PyTorch", "Transformers", "Audio Processing", "MFCC"],
    githubUrl: "https://github.com/SaiDhiren-Musaloji?tab=repositories",
    category: "ML Modeling"
  },
  {
    id: 8,
    title: "Self-Supervised Learning for User Localization",
    description: "• Traditional wireless localization methods require extensive labeled data and fail to extract meaningful representations from complex channel state information for accurate 3D position prediction\n\n• Implemented two-stage self-supervised approach using autoencoder for feature extraction followed by position prediction model, leveraging unlabeled channel data to achieve accurate localization with limited labeled samples.",
    image: "/images/projects/Self-Supervised Learning for User Localization.jpg",
    technologies: ["Python", "PyTorch", "Self-Supervised Learning", "Autoencoders", "Wireless ML"],
    githubUrl: "https://github.com/SaiDhiren-Musaloji?tab=repositories",
    category: "ML Modeling"
  },
  {
    id: 9,
    title: "Neural Machine Translation with Transformers",
    description: "• Traditional RNN-based translation systems struggle with long-range dependencies, parallel processing limitations, and maintaining contextual accuracy across variable-length sequences in English to Traditional Chinese translation\n\n• Implemented transformer encoder-decoder architecture with SentencePiece tokenization, custom beam search decoding, and label smoothing regularization achieving superior translation quality and reduced training time.",
    image: "/images/projects/bert.png",
    technologies: ["Python", "PyTorch", "Transformers", "SentencePiece", "NMT"],
    githubUrl: "https://github.com/SaiDhiren-Musaloji?tab=repositories",
    category: "Generative AI"
  },
  {
    id: 10,
    title: "Deep Neural Networks for Speech Classification",
    description: "• Traditional machine learning approaches fail to capture complex non-linear relationships in acoustic feature space and struggle with inherent variability in speech patterns across different speakers and contexts for phoneme classification\n\n• Implemented multi-layer neural network with context window concatenation, batch normalization, and advanced data augmentation techniques achieving substantial accuracy improvements over baseline methods.",
    image: "/images/projects/Deep Neural Networks for Speech Classification.webp",
    technologies: ["Python", "PyTorch", "Speech Processing", "MFCC", "Phoneme Classification"],
    githubUrl: "https://github.com/SaiDhiren-Musaloji?tab=repositories",
    category: "ML Modeling"
  },
  {
    id: 11,
    title: "Handwritten Telugu Character Recognition",
    description: "• Traditional OCR systems struggle with complex Indic scripts like Telugu due to character variations, ligatures, and contextual dependencies\n\n• Implemented CNN-based OCR framework with comprehensive data preprocessing, custom character segmentation, and multi-scale feature extraction achieving robust recognition accuracy for Telugu script.",
    image: "/images/projects/telugu handwritten recognition.png",
    technologies: ["TensorFlow", "Keras", "CNN", "OpenCV", "OCR"],
    githubUrl: "https://github.com/SaiDhiren-Musaloji?tab=repositories",
    category: "ML Modeling"
  },
  {
    id: 12,
    title: "Distributed Recognition Engine Using Cloud-Native Paradigms",
    description: "Problem: Traditional image and text recognition systems lack scalability and reliability for high-volume processing with varying workloads. Solution: Built asynchronous recognition pipeline using AWS EC2, SQS, S3, and Rekognition with stateless compute strategies, achieving 99.9% uptime through visibility timeout calibration and retry mechanisms.",
    image: "/images/projects/sqs.png",
    technologies: ["AWS", "Docker", "Kubernetes", "Terraform", "Cloud Computing"],
    githubUrl: "https://github.com/SaiDhiren-Musaloji?tab=repositories",
    category: "Engineering Solutions"
  },
  {
    id: 13,
    title: "Spatiotemporal Forecasting of Urban Traffic Networks",
    description: "Problem: Traditional traffic prediction models fail to capture complex spatiotemporal dependencies and topological interactions between network nodes in urban traffic systems. Solution: Engineered fusion architecture combining ARIMA, LSTM, and GNN models with PCA and t-SNE dimensionality reduction, achieving robust multi-horizon prediction across 325 sensors in the PEMS-BAY dataset.",
    image: "/images/projects/traffic_forecasting.png",
    technologies: ["Python", "ARIMA", "LSTM", "GNN", "PCA", "t-SNE", "Time Series"],
    githubUrl: "https://github.com/SaiDhiren-Musaloji?tab=repositories",
    category: "Data Analysis"
  },
  {
    id: 14,
    title: "High-Availability Financial Analytics System",
    description: "Problem: Traditional banking analytics systems lack real-time processing capabilities and fail to handle high concurrency requirements for fraud detection and transactional analysis. Solution: Developed transactional analytics platform with responsive dashboard, anomaly detection, and time-series mapping, containerized using Docker with CI/CD pipelines for high-availability deployment.",
    image: "/images/projects/database.png",
    technologies: ["Python", "Docker", "CI/CD", "Anomaly Detection", "Time Series", "Financial Analytics"],
    githubUrl: "https://github.com/SaiDhiren-Musaloji?tab=repositories",
    category: "Data Engineering"
  },
  {
    id: 15,
    title: "Parallelized ML Pipeline for Oenological Forecasting",
    description: "Problem: Traditional wine quality prediction models lack scalability and fail to handle large-scale datasets efficiently, resulting in slow training times and limited throughput for production inference. Solution: Created Spark-based distributed system on AWS EMR with partitioned data and in-memory caching, achieving 60% training speed improvement and high-throughput inference with autoscaling and performance logging.",
    image: "/images/projects/Parallel.png",
    technologies: ["Apache Spark", "AWS EMR", "ML Pipeline", "Distributed Computing", "Wine Quality Prediction"],
    githubUrl: "https://github.com/SaiDhiren-Musaloji?tab=repositories",
    category: "Engineering Solutions"
  },
  {
    id: 16,
    title: "Flight Data Analysis - Big Data Analytics & Performance Optimization",
    description: "Engineered scalable AWS-based MapReduce data processing pipelines using Apache Spark and Oozie orchestration, analyzing 22-year aviation dataset (1987-2008) containing 120M+ flight records with 40% improved query performance optimization. Implemented comprehensive scalability testing framework across varying data volumes (1GB to 500GB), demonstrating linear performance scalability and optimizing AWS resource allocation to reduce processing costs by 35%. Established high-performance aviation analytics platform supporting 15 concurrent users with sub-3-second query response times, delivering actionable insights on 5,000+ flight routes for executive decision-making processes.",
    image: "/images/projects/flight_data.png",
    technologies: ["AWS", "Apache Spark", "MapReduce", "Apache Oozie", "Big Data", "Data Analysis", "Scalability Testing", "Performance Optimization"],
    githubUrl: "https://github.com/SaiDhiren-Musaloji?tab=repositories",
    category: "Data Analysis"
  },
  {
    id: 17,
    title: "Equity Portfolio Management - Financial Analytics & Quantitative Analysis",
    description: "Orchestrated comprehensive portfolio optimization analysis using Python (Pandas, NumPy, Matplotlib) to evaluate $5M equity portfolio, implementing advanced risk-return modeling that generated 18% annual return improvement over S&P 500 benchmark. Directed quantitative analysis team of 3 members in developing algorithmic position sizing strategies, utilizing simulations across 500+ securities to reduce portfolio volatility by 23% while maintaining 12.4% target return threshold. Architected automated performance tracking system monitoring 15 key performance indicators (KPIs) daily, enabling real-time investment decisions that improved Sharpe ratio from 1.2 to 1.8 over 12-month analytical period.",
    image: "/images/projects/portfolio.png",
    technologies: ["Python", "Pandas", "NumPy", "Matplotlib", "Quantitative Analysis", "Risk Modeling", "Algorithmic Trading", "Portfolio Optimization", "Sharpe Ratio"],
    githubUrl: "https://github.com/SaiDhiren-Musaloji?tab=repositories",
    category: "Engineering Solutions"
  },
  {
    id: 18,
    title: "Banking Transaction Management System - Agile Full-Stack RDBMS Development & Analytics",
    description: "Engineered enterprise-grade relational database management system integrating MySQL primary databases with Snowflake data warehouse and DynamoDB document storage for banking network infrastructure, designing normalized schema with 15+ tables to handle 10,000+ daily transactions with 99.9% data integrity and sub-200ms query response times. Developed user-centric web interface using Flask and REST APIs to support comprehensive transaction processing workflows, implementing real-time balance updates and transaction history tracking for 500+ concurrent users based on stakeholder requirements. Implemented agile development methodology across 2-sprint delivery cycle, conducting user story analysis and stakeholder requirement gathering to deliver full-stack banking solution with transaction management, user authentication, and reporting capabilities.",
    image: "/images/projects/database.png",
    technologies: ["MySQL", "Snowflake", "DynamoDB", "Flask", "REST APIs", "Python", "Agile Development", "Database Design", "Web Interface", "Transaction Management"],
    githubUrl: "https://github.com/SaiDhiren-Musaloji?tab=repositories",
    category: "Data Engineering"
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
  title: "AI Data Engineer & Data Scientist",
  location: "New Jersey, USA",
  education: "MS in Data Science - New Jersey Institute of Technology (GPA: 3.85/4.0)",
  experience: "AI Data Engineer Intern at Tech Mahindra",
  about: "Passionate AI Data Engineer and Data Scientist with expertise in machine learning, deep learning, and cloud computing. Recent graduate with MS in Data Science from NJIT with a strong focus on transformer models, reinforcement learning, and ethical AI practices. Experienced in developing scalable solutions, building end-to-end data pipelines, and contributing to national AI initiatives.",
  aboutDetailed: "My journey into data science began with a curiosity about how machines can learn and make intelligent decisions. Coming from an Electronics and Communication Engineering background, I discovered the fascinating intersection of mathematics, programming, and human cognition that defines modern AI.\n\nWhat drives me is the potential of AI to solve real-world problems that impact people's lives. During my time at Tech Mahindra's Makers Lab as an AI Data Engineer Intern, I contributed to Project Indus, a national initiative to develop Large Language Models for Indian languages, while building comprehensive data pipelines and NLP workflows.\n\nI believe in the power of continuous learning and experimentation. My portfolio reflects this philosophy - from implementing adversarial attacks to understand AI vulnerabilities, to building RAG systems for intelligent research assistance, and developing enterprise-grade data solutions."
};

export const experience = [
  {
    id: 1,
    title: "Data Engineer",
    company: "TAWIN Solutions LLC",
    location: "Dallas, TX",
    duration: "May 2025 – Current",
    description: [
      "Architected scalable schemas across Azure SQL, Synapse, and on‑prem SQL Server to strengthen BI foundations and analytics capabilities",
      "Built robust ETL/ELT pipelines with Azure Data Factory and SSIS to integrate heterogeneous sources into the EDW with quality and lineage",
      "Delivered ML solutions for predictive pricing using Azure ML with MLOps via Azure DevOps, Azure Monitor, versioning, and automated retraining",
      "Modeled dimensional (star/snowflake) and relational schemas in Synapse enabling self‑service BI across business units",
      "Optimized performance with query tuning, indexing, and stored procedure refactors across Azure SQL and SQL Server",
      "Leveraged AWS EC2, RDS, and S3 for dev/test compute, managed databases, and data lake storage/backup",
      "Implemented security and governance via Azure Key Vault and Azure Purview for credential management and lineage",
      "Partnered with stakeholders to translate complex requirements into actionable technical specifications and solutions",
      "Drove cross‑functional collaboration among analysts, DBAs, developers, and data scientists for integrated BI delivery",
      "Established technical documentation standards: architecture diagrams, data dictionaries, ETL docs, and runbooks",
      "Developed advanced SQL: complex queries, stored procedures, UDFs, views, and triggers for automation and real‑time reporting",
      "Implemented CI/CD with Azure DevOps for automated deployment of database objects and ETL packages"
    ]
  },
  {
    id: 2,
    title: "AI Data Engineer Intern",
    company: "Tech Mahindra, Makers Lab",
    location: "Pune, India",
    duration: "Oct 2023 – Jan 2024",
    description: [
      "Implemented ADF pipelines to ETL data from Snowflake, DynamoDB, PostgreSQL, and Oracle; processed 500K+ multilingual texts with 99.2% quality",
      "Engineered NLP preprocessing with Python and Spark (tokenization, regex filtering, normalization) improving accuracy by 25% across 8 languages",
      "Built automated data quality checks with statistical outlier detection; reduced manual cleaning by 60% and improved content quality by 20%",
      "Collaborated to document and deploy production ML models; CI/CD reduced deployment time from hours to minutes with 99.8% uptime"
    ]
  }
  ,
  {
    id: 3,
    title: "Cloud Computing Intern",
    company: "LTIMindtree",
    location: "Hyderabad, India",
    duration: "Feb 2023 - Apr 2023",
    description: [
      "Studied cloud fundamentals with emphasis on AWS core services",
      "Accelerated data workflows using EC2, S3, Lambda, VPC, and RDS; used Docker, Kubernetes, Terraform",
      "Assisted cloud migration and big data initiatives improving resource use and cost efficiency",
      "Gained hands‑on IaC, container orchestration, and DevOps best practices"
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
  { id: "all", name: "All Projects", count: 18 },
  { id: "data-analysis", name: "Data Analysis", count: 3 },
  { id: "data-engineering", name: "Data Engineering", count: 2 },
  { id: "ml-modeling", name: "ML Modeling", count: 7 },
  { id: "engineering-solutions", name: "Engineering Solutions", count: 3 },
  { id: "generative-ai", name: "Generative AI", count: 3 }
]; 