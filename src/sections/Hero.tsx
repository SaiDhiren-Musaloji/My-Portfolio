import React from 'react';
import { personalInfo, heroContent } from '../data/portfolioData';
import './Hero.css';

const Hero: React.FC = () => {
  const scrollToSection = (sectionId: string) => {
    document.getElementById(sectionId)?.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <section id="hero" className="hero">
      <div className="hero-bg" aria-hidden="true">
        <div className="hero-aurora hero-aurora--1" />
        <div className="hero-aurora hero-aurora--2" />
        <div className="hero-aurora hero-aurora--3" />
      </div>

      <div className="hero-container">
        <div className="hero-layout">
          <div className="hero-main">
            <p className="hero-role">
              <span className="hero-role-dot" aria-hidden="true" />
              {heroContent.role}
            </p>

            <h1 className="hero-name">{personalInfo.name}</h1>
            <p className="hero-headline">{heroContent.headline}</p>

            <div className="hero-block">
              <span className="hero-label">Building now</span>
              <div className="hero-work-list">
                {heroContent.currentWork.map((item) => (
                  <button
                    key={item.projectId}
                    type="button"
                    className="hero-work-item"
                    onClick={() => scrollToSection('projects')}
                  >
                    <span className="hero-work-desc">{item.description}</span>
                    <span className="hero-work-name">{item.label}</span>
                  </button>
                ))}
              </div>
            </div>

            <div className="hero-cta">
              <button
                type="button"
                className="hero-btn-primary"
                onClick={() => scrollToSection('projects')}
              >
                View Projects
              </button>
              <button
                type="button"
                className="hero-btn-link"
                onClick={() => scrollToSection('contact')}
              >
                Get in touch
              </button>
            </div>
          </div>

          <div className="hero-visual">
            <div className="hero-photo-frame">
              <div className="hero-blob hero-blob--1" aria-hidden="true" />
              <div className="hero-blob hero-blob--2" aria-hidden="true" />
              <div className="hero-blob hero-blob--3" aria-hidden="true" />
              <img
                src="/images/profile.png"
                alt={personalInfo.name}
                className="hero-photo"
              />
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Hero;
