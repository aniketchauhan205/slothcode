// frontend/src/store/useProjectStore.ts
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

// 1. Define the shape of your state
interface ProjectState {
  files: Record<string, string>; // Maps file paths to their content
  setFile: (path: string, content: string) => void;
  clearProject: () => void;
}

// 2. Create the store with the type
export const useProjectStore = create<ProjectState>()(
  persist(
    (set) => ({
      files: {},
      setFile: (path, content) =>
        set((state) => ({
          files: { ...state.files, [path]: content }
        })),
      clearProject: () => set({ files: {} }),
    }),
    { name: 'project-storage' }
  )
);