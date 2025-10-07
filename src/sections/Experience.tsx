import React, { useState } from 'react';
import { experience } from '../data/portfolioData';
import './Experience.css';

const Experience: React.FC = () => {
  const [expandedIds, setExpandedIds] = useState<number[]>([]);

  const toggleExpand = (id: number) => {
    setExpandedIds(prev => (
      prev.includes(id) ? prev.filter(expandedId => expandedId !== id) : [...prev, id]
    ));
  };

  const getVisibleDescription = (id: number, items: string[]) => {
    const isExpanded = expandedIds.includes(id);
    return isExpanded ? items : [];
  };

  return (
    <section id="experience" className="experience">
      <div className="experience-container">
        <div className="section-header">
          <h2 className="section-title">Professional Experience</h2>
          <div className="section-divider"></div>
        </div>
        
        <div className="experience-content">
          <div className="experience-timeline">
            {experience.map((exp) => {
              const isExpanded = expandedIds.includes(exp.id);
              const visibleDescription = getVisibleDescription(exp.id, exp.description);
              return (
              <div key={exp.id} className="experience-item">
                <div className="experience-marker"></div>
                <div
                  className="experience-card"
                  onClick={() => toggleExpand(exp.id)}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      toggleExpand(exp.id);
                    }
                  }}
                >
                  <div className="experience-header">
                    <h3 className="experience-title">{exp.title}</h3>
                    <div className="experience-meta">
                      <span className="company">{exp.company}</span>
                      <span className="location">{exp.location}</span>
                      <span className="duration">{exp.duration}</span>
                    </div>
                  </div>
                  {visibleDescription.length > 0 && (
                    <ul className="experience-description">
                      {visibleDescription.map((item, index) => (
                        <li key={index}>{item}</li>
                      ))}
                    </ul>
                  )}
                  <div className="experience-hint" aria-hidden="true">
                    {isExpanded ? 'Hide details' : 'View role details'}
                    <span className={`chevron ${isExpanded ? 'up' : 'down'}`}></span>
                  </div>
                </div>
              </div>
              );
            })}
          </div>
        </div>
      </div>
    </section>
  );
};

export default Experience; 