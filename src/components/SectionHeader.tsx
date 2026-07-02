import React from 'react';
import './SectionHeader.css';

interface SectionHeaderProps {
  eyebrow: string;
  title: string;
  subtitle?: string;
  align?: 'left' | 'center';
  light?: boolean;
}

const SectionHeader: React.FC<SectionHeaderProps> = ({
  eyebrow,
  title,
  subtitle,
  align = 'center',
  light = false,
}) => (
  <div className={`section-header-v2 ${align} ${light ? 'light' : ''}`}>
    <span className="section-eyebrow">{eyebrow}</span>
    <h2 className="section-heading">{title}</h2>
    {subtitle && <p className="section-subtitle">{subtitle}</p>}
    <div className="section-line" />
  </div>
);

export default SectionHeader;
