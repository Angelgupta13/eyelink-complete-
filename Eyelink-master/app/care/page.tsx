import { Suspense } from "react";
import CarePageClient from "./CarePageClient";

interface PageProps {
  searchParams: Promise<{ [key: string]: string | string[] | undefined }>;
}

export default async function CarePage({ searchParams }: PageProps) {
  const resolvedParams = await searchParams;

  return (
    <Suspense fallback={<div>Loading...</div>}>
      <CarePageClient initialSearchParams={resolvedParams} />
    </Suspense>
  );
}
