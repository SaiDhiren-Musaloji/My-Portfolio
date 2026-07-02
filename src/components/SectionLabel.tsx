import React from 'react';
import './SectionLabel.css';

interface SectionLabelProps {
  index: string;
  title: string;
  description?: string;
  light?: boolean;
}

const SectionLabel: React.FC<SectionLabelProps> = ({ index, title, description, light }) => (
  <div className={`section-label${light ? ' section-label--light' : ''}`}>
    <div className="section-label-top">
      <span className="section-label-index">{index}</span>
      <h2 className="section-label-title">{title}</h2>
    </div>
    {description && <p className="section-label-desc">{description}</p>}
  </div>
);

export default SectionLabel;
