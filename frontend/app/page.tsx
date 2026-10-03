import Workspace from "@/components/Workspace";
import { DetailsProvider } from "@/lib/details";

export default function Home() {
  return (
    <DetailsProvider>
      <Workspace />
    </DetailsProvider>
  );
}
