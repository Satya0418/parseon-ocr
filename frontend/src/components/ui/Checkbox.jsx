const Checkbox = ({ label, ...props }) => {
  return (
    <label className="flex items-center gap-2 cursor-pointer group">
      <div className="relative flex items-center">
        <input
          type="checkbox"
          className="peer appearance-none w-5 h-5 border-2 border-secondary rounded-md checked:bg-accent checked:border-accent transition-all cursor-pointer"
          {...props}
        />
        <svg
          className="absolute w-5 h-5 pointer-events-none hidden peer-checked:block text-white p-0.5"
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="4"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <polyline points="20 6 9 17 4 12"></polyline>
        </svg>
      </div>
      {label && <span className="text-sm text-primary/80 font-medium group-hover:text-primary transition-colors">{label}</span>}
    </label>
  );
};

export default Checkbox;
