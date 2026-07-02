import React from 'react';
import { contactInfo, personalInfo } from '../data/portfolioData';
import './Footer.css';

const Footer: React.FC = () => (
  <footer className="site-footer">
    <div className="page-wrap site-footer-inner">
      <span className="site-footer-copy">
        © {new Date().getFullYear()} {personalInfo.name}
      </span>
      <div className="site-footer-links">
        <a href={contactInfo.github} target="_blank" rel="noopener noreferrer">GitHub</a>
        <a href={contactInfo.linkedin} target="_blank" rel="noopener noreferrer">LinkedIn</a>
        <a href={`mailto:${contactInfo.email}`}>Email</a>
      </div>
    </div>
  </footer>
);

export default Footer;
