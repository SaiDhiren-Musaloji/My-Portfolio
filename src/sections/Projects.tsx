import React, { useState } from 'react';
import { projects, projectCategories, featuredProjectIds } from '../data/portfolioData';
import ProjectModal from '../components/ProjectModal';
import SectionLabel from '../components/SectionLabel';
import './Projects.css';

const Projects: React.FC = () => {
  const [activeFilter, setActiveFilter] = useState('all');
  const [selectedProject, setSelectedProject] = useState<number | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  const featured = projects.filter((p) => featuredProjectIds.includes(p.id));

  const matchesCategory = (catId: string, p: typeof projects[number]) => {
    if (catId === 'all') return true;
    const legacy = p.category?.toLowerCase().replace(/\s+/g, '-');
    return p.categories?.includes(catId) || p.primaryCategory === catId || legacy === catId;
  };

  const archive = projects.filter((p) => {
    if (featuredProjectIds.includes(p.id)) return false;
    return matchesCategory(activeFilter, p);
  });

  const openProject = (id: number) => {
    setSelectedProject(id);
    setIsModalOpen(true);
  };

  const closeModal = () => {
    setIsModalOpen(false);
    setSelectedProject(null);
  };

  const selectedData = selectedProject ? projects.find((p) => p.id === selectedProject) || null : null;

  const filters = projectCategories.filter((c) => c.id !== 'all');

  return (
    <section id="projects" className="page-section page-section--surface">
      <div className="page-wrap">
        <SectionLabel
          index="01"
          title="Work"
          description="Production systems at MakersLab, plus research and engineering projects."
        />

        {/* Featured row */}
        <div className="work-featured">
          {featured.map((p, i) => (
            <button
              key={p.id}
              type="button"
              className={`work-card work-card--${i + 1}`}
              onClick={() => openProject(p.id)}
            >
              <div className="work-card-num">0{i + 1}</div>
              <div className="work-card-body">
                <span className="work-card-tag">MakersLab</span>
                <h3 className="work-card-title">{p.title}</h3>
                <div className="work-card-tech">
                  {p.technologies.slice(0, 3).map((t) => (
                    <span key={t}>{t}</span>
                  ))}
                </div>
              </div>
              <span className="work-card-arrow">→</span>
            </button>
          ))}
        </div>

        {/* Archive */}
        <div className="work-archive">
          <div className="work-filters">
            <button
              type="button"
              className={`work-filter ${activeFilter === 'all' ? 'active' : ''}`}
              onClick={() => setActiveFilter('all')}
            >
              All
            </button>
            {filters.map((f) => (
              <button
                key={f.id}
                type="button"
                className={`work-filter ${activeFilter === f.id ? 'active' : ''}`}
                onClick={() => setActiveFilter(f.id)}
              >
                {f.name}
              </button>
            ))}
          </div>

          <div className="work-list">
            {archive.map((p) => (
              <button
                key={p.id}
                type="button"
                className="work-row"
                onClick={() => openProject(p.id)}
              >
                <span className="work-row-title">{p.title}</span>
                <span className="work-row-tags">
                  {p.technologies.slice(0, 3).join(' · ')}
                </span>
                <span className="work-row-arrow">→</span>
              </button>
            ))}
            {archive.length === 0 && (
              <p className="work-empty">No projects in this category.</p>
            )}
          </div>
        </div>
      </div>

      <ProjectModal project={selectedData} isOpen={isModalOpen} onClose={closeModal} />
    </section>
  );
};

export default Projects;
