import React from 'react';
import { experience } from '../data/portfolioData';
import './Experience.css';

const Experience: React.FC = () => {
  return (
    <section id="experience" className="experience">
      <div className="experience-container">
        <div className="section-header">
          <h2 className="section-title">Professional Experience</h2>
          <div className="section-divider"></div>
        </div>
        
        <div className="experience-content">
          <div className="experience-timeline">
            {experience.map(exp => (
              <div key={exp.id} className="experience-item">
                <div className="experience-marker"></div>
                <div className="experience-card">
                  <div className="experience-header">
                    <h3 className="experience-title">{exp.title}</h3>
                    <div className="experience-meta">
                      <span className="company">{exp.company}</span>
                      <span className="location">{exp.location}</span>
                      <span className="duration">{exp.duration}</span>
                    </div>
                  </div>
                  <ul className="experience-description">
                    {exp.description.map((item, index) => (
                      <li key={index}>{item}</li>
                    ))}
                  </ul>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

export default Experience; 