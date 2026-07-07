import React from 'react';
import { skillDomains } from '../data/portfolioData';
import SectionLabel from '../components/SectionLabel';
import './Skills.css';

const Skills: React.FC = () => (
  <section id="skills" className="page-section stack-section">
    <div className="stack-section-bg" aria-hidden="true" />
    <div className="page-wrap">
      <SectionLabel
        index="02"
        title="Stack"
        description="Core tools for building and shipping AI systems."
        light
      />

      <div className="stack-grid">
        {skillDomains.map((domain) => (
          <div key={domain.id} className="stack-glass">
            <h3 className="stack-glass-title">{domain.title}</h3>
            <p className="stack-glass-desc">{domain.description}</p>
            <ul className="stack-skill-list">
              {domain.skills.map((skill) => (
                <li key={skill.name} className="stack-skill-tag">
                  {skill.name}
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </div>
  </section>
);

export default Skills;
