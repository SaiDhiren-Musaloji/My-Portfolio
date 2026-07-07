import React, { useState } from 'react';
import { experience, education, certificates } from '../data/portfolioData';
import SectionLabel from '../components/SectionLabel';
import './Career.css';

const Career: React.FC = () => {
  const [expandedId, setExpandedId] = useState<number | null>(1);
  const [certModal, setCertModal] = useState<typeof certificates[0] | null>(null);

  return (
    <section id="career" className="page-section page-section--surface">
      <div className="page-wrap">
        <SectionLabel
          index="03"
          title="Career"
          singleLine
          description="Where I've built and shipped AI systems."
        />

        {/* Experience timeline */}
        <div className="career-jobs">
          {experience.map((job) => {
            const open = expandedId === job.id;
            return (
              <div key={job.id} className={`career-job ${open ? 'open' : ''}`}>
                <button
                  type="button"
                  className="career-job-header"
                  onClick={() => setExpandedId(open ? null : job.id)}
                >
                  <div className="career-job-left">
                    <span className="career-job-period">{job.duration}</span>
                    <h3 className="career-job-title">{job.title}</h3>
                    <span className="career-job-co">{job.company} · {job.location}</span>
                  </div>
                  <span className="career-job-toggle">{open ? '−' : '+'}</span>
                </button>
                {open && (
                  <ul className="career-job-details">
                    {job.description.map((d, i) => (
                      <li key={i}>{d}</li>
                    ))}
                  </ul>
                )}
              </div>
            );
          })}
        </div>

        {/* Education strip */}
        <div className="career-edu">
          {education.map((edu) => (
            <div key={edu.id} className="career-edu-item">
              <span className="career-edu-degree">{edu.degree}</span>
              <span className="career-edu-school">{edu.school}</span>
              <span className="career-edu-meta">{edu.duration} · GPA {edu.gpa}</span>
            </div>
          ))}
        </div>

        {/* Certificates row */}
        <div className="career-certs">
          <span className="career-certs-label">Certifications</span>
          <div className="career-certs-row">
            {certificates.map((cert) => (
              <button
                key={cert.id}
                type="button"
                className="career-cert"
                onClick={() => setCertModal(cert)}
              >
                <img src={cert.image} alt={cert.title} />
                <span>{cert.title}</span>
              </button>
            ))}
          </div>
        </div>
      </div>

      {certModal && (
        <div className="cert-overlay" onClick={() => setCertModal(null)}>
          <div className="cert-modal" onClick={(e) => e.stopPropagation()}>
            <button type="button" className="cert-close" onClick={() => setCertModal(null)}>×</button>
            <img src={certModal.image} alt={certModal.title} />
            <p className="cert-modal-title">{certModal.title}</p>
            <p className="cert-modal-meta">{certModal.issuer} · {certModal.issueDate}</p>
          </div>
        </div>
      )}
    </section>
  );
};

export default Career;
