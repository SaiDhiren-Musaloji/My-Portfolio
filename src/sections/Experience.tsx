import React, { useState } from 'react';
import { experience } from '../data/portfolioData';
import SectionHeader from '../components/SectionHeader';
import Tile3D from '../components/Tile3D';
import './Experience.css';

const Experience: React.FC = () => {
  const [expandedIds, setExpandedIds] = useState<number[]>([1]);

  const toggleExpand = (id: number) => {
    setExpandedIds((prev) =>
      prev.includes(id) ? prev.filter((i) => i !== id) : [...prev, id]
    );
  };

  return (
    <section id="experience" className="experience">
      <div className="experience-container">
        <SectionHeader
          eyebrow="Career"
          title="Where I've Built"
          subtitle="From enterprise data pipelines to production agentic AI — a track record of shipping real systems."
        />

        <div className="experience-grid">
          {experience.map((exp, index) => {
            const isExpanded = expandedIds.includes(exp.id);
            const isCurrent = index === 0;

            return (
              <Tile3D
                key={exp.id}
                className={`exp-tile ${isCurrent ? 'exp-tile--current' : ''} clickable`}
                intensity={8}
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
                <div className="exp-card">
                  <div className="exp-card-top">
                    <div className="exp-card-meta">
                      {isCurrent && <span className="exp-current-badge">Current</span>}
                      <span className="exp-duration">{exp.duration}</span>
                    </div>
                    <h3 className="exp-title">{exp.title}</h3>
                    <div className="exp-company-row">
                      <span className="exp-company">{exp.company}</span>
                      <span className="exp-location">{exp.location}</span>
                    </div>
                  </div>

                  {isExpanded && (
                    <ul className="exp-bullets">
                      {exp.description.map((item, i) => (
                        <li key={i}>{item}</li>
                      ))}
                    </ul>
                  )}

                  <div className="exp-toggle-hint">
                    {isExpanded ? 'Collapse ↑' : 'Expand details ↓'}
                  </div>
                </div>
              </Tile3D>
            );
          })}
        </div>
      </div>
    </section>
  );
};

export default Experience;
