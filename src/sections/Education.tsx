import React from 'react';
import { education } from '../data/portfolioData';
import './Education.css';

const Education: React.FC = () => {
  return (
    <section id="education" className="education">
      <div className="education-container">
        <div className="section-header">
          <h2 className="section-title">Education</h2>
          <div className="section-divider"></div>
        </div>
        
        <div className="education-content">
          <div className="education-grid">
            {education.map(edu => (
              <div key={edu.id} className="education-card">
                <div className="education-header">
                  <h3 className="degree">{edu.degree}</h3>
                  <h4 className="school">{edu.school}</h4>
                  <div className="education-meta">
                    <span className="duration">{edu.duration}</span>
                    <span className="gpa">GPA: {edu.gpa}</span>
                  </div>
                </div>
                
                {edu.coursework && (
                  <div className="coursework">
                    <h5>Key Coursework:</h5>
                    <p className="coursework-text">
                      {edu.coursework.join(', ')}
                    </p>
                  </div>
                )}
                
                {edu.capstone && (
                  <div className="capstone">
                    <h5>Capstone Research:</h5>
                    <p className="capstone-text">{edu.capstone}</p>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

export default Education; 