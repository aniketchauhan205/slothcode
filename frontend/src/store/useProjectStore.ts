import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { Job, JobEvent } from '../types';

interface ProjectState {
  jobsById: Record<string, Job>;
  eventsByJob: Record<string, JobEvent[]>;
  filesByJob: Record<string, Record<string, string>>;
  setJob: (job: Job) => void;
  updateJob: (jobId: string, patch: Partial<Job>) => void;
  addEvent: (jobId: string, event: JobEvent) => void;
  setFile: (jobId: string, path: string, content: string) => void;
  setFiles: (jobId: string, files: Record<string, string>) => void;
  clearProject: (jobId: string) => void;
}

export const useProjectStore = create<ProjectState>()(
  persist(
    (set) => ({
      jobsById: {},
      eventsByJob: {},
      filesByJob: {},
      setJob: (job) =>
        set((state) => ({
          jobsById: {
            ...state.jobsById,
            [job.id]: job,
          },
        })),
      updateJob: (jobId, patch) =>
        set((state) => ({
          jobsById: {
            ...state.jobsById,
            ...(state.jobsById[jobId]
              ? {
                  [jobId]: {
                    ...state.jobsById[jobId],
                    ...patch,
                  },
                }
              : {}),
          },
        })),
      addEvent: (jobId, event) =>
        set((state) => {
          const existing = state.eventsByJob[jobId] ?? [];
          const alreadyExists = existing.some(
            (item) => item.timestamp === event.timestamp && item.type === event.type,
          );
          if (alreadyExists) return {};
          return {
            eventsByJob: {
              ...state.eventsByJob,
              [jobId]: [...existing, event],
            },
          };
        }),
      setFile: (jobId, path, content) =>
        set((state) => ({
          filesByJob: {
            ...state.filesByJob,
            [jobId]: {
              ...(state.filesByJob[jobId] ?? {}),
              [path]: content,
            },
          },
        })),
      setFiles: (jobId, files) =>
        set((state) => ({
          filesByJob: {
            ...state.filesByJob,
            [jobId]: {
              ...(state.filesByJob[jobId] ?? {}),
              ...files,
            },
          },
        })),
      clearProject: (jobId) =>
        set((state) => {
          const { [jobId]: _removedJob, ...jobsById } = state.jobsById;
          const { [jobId]: _removedEvents, ...eventsByJob } = state.eventsByJob;
          const { [jobId]: _removed, ...rest } = state.filesByJob;
          return { jobsById, eventsByJob, filesByJob: rest };
        }),
    }),
    { name: 'project-storage' }
  )
);
