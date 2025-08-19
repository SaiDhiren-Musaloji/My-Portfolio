export interface Project {
  id: number;
  title: string;
  description: string;
  image: string;
  technologies: string[];
  githubUrl?: string;
  liveUrl?: string;
  category?: string;
  // Additional details for modal
  detailedDescription?: string;
  challenges?: string[];
  solutions?: string[];
  keyFeatures?: string[];
  results?: string[];
  duration?: string;
  teamSize?: string;
  role?: string;
}

export interface Certificate {
  id: number;
  title: string;
  issuer: string;
  image: string;
  pdfUrl?: string;
  issueDate: string;
}

export interface Skill {
  id: number;
  name: string;
  category: 'machine-learning' | 'deep-learning' | 'data-processing' | 'cloud-computing' | 'programming' | 'tools';
  proficiency: number; // 1-100
}

export interface ContactInfo {
  email: string;
  linkedin: string;
  github: string;
  phone?: string;
}

export interface Experience {
  id: number;
  title: string;
  company: string;
  location: string;
  duration: string;
  description: string[];
}

export interface Education {
  id: number;
  degree: string;
  school: string;
  duration: string;
  gpa: string;
  coursework?: string[];
  capstone?: string;
}

export interface ProjectCategory {
  id: string;
  name: string;
  count: number;
} 