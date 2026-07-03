import React from 'react';
import './SectionLabel.css';

interface SectionLabelProps {
  index: string;
  title: string;
  description?: React.ReactNode;
  light?: boolean;
  singleLine?: boolean;
}

const SectionLabel: React.FC<SectionLabelProps> = ({
  index,
  title,
  description,
  light,
  singleLine,
}) => (
  <div className={`section-label${light ? ' section-label--light' : ''}`}>
    <div className="section-label-top">
      <span className="section-label-index">{index}</span>
      <h2 className="section-label-title">{title}</h2>
    </div>
    {description && (
      <p className={`section-label-desc${singleLine ? ' section-label-desc--single-line' : ''}`}>
        {description}
      </p>
    )}
  </div>
);

export default SectionLabel;
