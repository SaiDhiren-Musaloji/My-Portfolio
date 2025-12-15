import React from 'react';
import { personalInfo } from '../data/portfolioData';
import './About.css';

const About: React.FC = () => {
  // Split the detailed about text into paragraphs
  const aboutParagraphs = personalInfo.aboutDetailed.split('\n\n');

  return (
    <section id="about" className="about">
      <div className="about-container">
        <div className="section-header">
          <h2 className="section-title">About Me</h2>
          <div className="section-divider"></div>
        </div>
        
        <div className="about-content">
          <div className="about-text">
            {aboutParagraphs.map((paragraph, index) => (
              <p key={index} className="about-description">
                {paragraph}
              </p>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

export default About; 