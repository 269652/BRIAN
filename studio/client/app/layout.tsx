import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Brian Studio",
  description: "Visual language model editor — build, compose, train and deploy",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body style={{ height: "100vh", overflow: "hidden" }}>{children}</body>
    </html>
  );
}
