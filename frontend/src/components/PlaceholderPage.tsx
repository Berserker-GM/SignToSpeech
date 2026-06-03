type Props = { title: string; description: string };

export function PlaceholderPage({ title, description }: Props) {
  return (
    <div className="content placeholder-page">
      <h2>{title}</h2>
      <p>{description}</p>
    </div>
  );
}
